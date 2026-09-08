#!/usr/bin/env python3
import math
import numpy as np

import cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog
from dragonpilot.selfdrive.controls.lib.acm import ACM
from dragonpilot.selfdrive.controls.lib.aem import AEM
from dragonpilot.selfdrive.controls.lib.dtsc import DTSC

# ====================== 可調參數區（TUNING PARAMS） ======================

EARLYNESS = 1.0   
STRENGTH  = 0.65
SENS      = 1.50   

SLEW_V_BP = [0., 11.1, 19.4, 25.0] 
# 🌟 衝突解除 1：放寬低速域的變化率，允許系統瞬間執行「克隆指令」，消除物理遲鈍
ACCEL_SLEW_RATE_BP = [2.5, 2.0, 1.0, 0.4] 
DECEL_SLEW_RATE_BP = [3.0, 2.5, 2.0, 1.5]

ACCEL_CLIP_FAST_LEAD_DECEL_THRESH = -0.2       

ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5

HARD_MIN_LEAD_DIST_M = 3.0 
CLOSING_MIN_MPS = 0.6 
A_REQ_DIST_BUFFER_M = 2.0
PREBRAKE_DIST_MIN_M = 1.5 
PREBRAKE_DIST_MAX_M_BASE = 30.0  
LEAD_SLOW_MPS_BASE = 9.0  
TTC_START_BASE = 2.2
TTC_FULL_BASE  = 1.2
A_REQ_START_BASE = -1.2
A_REQ_FULL_BASE  = -2.6
PREBRAKE_MAX_DECEL_BASE = -0.65 

# =======================================================================

LON_MPC_STEP = 0.2

A_CRUISE_MAX_VALS = [1.25,  1.15,   1.05,   0.8,   0.644,  0.441,  0.198] 
A_CRUISE_MAX_BP   = [0.0,   2.78,   8.33,   15.0,  20.0,   25.0,   30.0]

CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]

_A_TOTAL_MAX_V = [2.5, 3.8] 
_A_TOTAL_MAX_BP = [20., 40.]

class DPFlags:
  ACM = 1
  AEM = 2
  DTSC = 2 ** 2
  pass

def smooth_interp(x, xp, fp):
  x = np.clip(x, xp[0], xp[-1])
  idx = np.searchsorted(xp, x, side='right') - 1
  if idx >= len(xp) - 1: return float(fp[-1])
  if idx < 0: return float(fp[0])
  x0, x1 = xp[idx], xp[idx+1]
  y0, y1 = fp[idx], fp[idx+1]
  mu = (x - x0) / max((x1 - x0), 1e-6)
  mu_smooth = (1.0 - math.cos(mu * math.pi)) / 2.0
  return float(y0 * (1.0 - mu_smooth) + y1 * mu_smooth)

def get_max_accel(v_ego):
  return smooth_interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3

def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.0))
  return [a_target[0], min(a_target[1], a_x_allowed)]

def _get_lead_decel_a(sm) -> float:
  try:
    if sm['radarState'].leadOne.status:
      return float(sm['radarState'].leadOne.aLeadK)
  except Exception:
    pass
  return 0.0

def _pick_closest_lead(radarstate):
  best = None
  best_d = 1e9
  for lead in (radarstate.leadOne, radarstate.leadTwo):
    try:
      if lead is not None and lead.status and np.isfinite(lead.dRel):
        if lead.dRel < best_d:
          best_d = float(lead.dRel)
          best = lead
    except Exception:
      pass
  return best

def _lead_metrics(v_ego: float, radarstate):
  lead = _pick_closest_lead(radarstate)
  if lead is None:
    return None, float('inf'), v_ego, 0.0, float('inf'), 0.0
  d_rel = float(getattr(lead, 'dRel', 1e9))
  v_lead = float(getattr(lead, 'vLead', v_ego))
  closing = float(v_ego - v_lead)
  if (not np.isfinite(d_rel)) or d_rel <= 0.0: ttc = 0.0
  elif closing <= CLOSING_MIN_MPS: ttc = float('inf')
  else: ttc = d_rel / max(closing, 1e-3)
  d_eff = max(d_rel - A_REQ_DIST_BUFFER_M, 0.5)
  if closing <= CLOSING_MIN_MPS: a_req = 0.0
  else: a_req = - (closing * closing) / (2.0 * d_eff)
  return lead, d_rel, v_lead, closing, float(ttc), float(a_req)

def _derived_thresholds():
  ttc_start = TTC_START_BASE + 0.9 * (EARLYNESS - 1.0)
  ttc_full  = TTC_FULL_BASE  + 0.5 * (EARLYNESS - 1.0)
  ttc_start = float(np.clip(ttc_start, 1.6, 4.0))
  ttc_full  = float(np.clip(ttc_full, 0.8, ttc_start - 0.2))
  a_req_start = A_REQ_START_BASE + 0.7 * (EARLYNESS - 1.0) + 0.5 * (SENS - 1.0)
  a_req_full  = A_REQ_FULL_BASE  + 0.5 * (EARLYNESS - 1.0) + 0.3 * (SENS - 1.0)
  a_req_start = float(np.clip(a_req_start, -2.0, -0.3))
  a_req_full  = float(np.clip(a_req_full,  -4.5, a_req_start - 0.3))
  dist_max = PREBRAKE_DIST_MAX_M_BASE * (0.9 + 0.35 * (SENS - 1.0)) * (0.95 + 0.25 * (EARLYNESS - 1.0))
  dist_max = float(np.clip(dist_max, 35.0, 110.0))
  lead_slow_mps = LEAD_SLOW_MPS_BASE * (0.95 + 0.25 * (SENS - 1.0)) * (0.95 + 0.15 * (EARLYNESS - 1.0))
  lead_slow_mps = float(np.clip(lead_slow_mps, 6.0, 14.0))
  prebrake_max_decel = PREBRAKE_MAX_DECEL_BASE * STRENGTH
  prebrake_max_decel = float(np.clip(prebrake_max_decel, -3.5, -0.8))
  return ttc_start, ttc_full, a_req_start, a_req_full, dist_max, lead_slow_mps, prebrake_max_decel

def _approach_trigger(v_ego: float, radarstate):
  ttc_start, ttc_full, a_req_start, a_req_full, dist_max, lead_slow_mps, prebrake_max_decel = _derived_thresholds()
  lead, d_rel, v_lead, closing, ttc, a_req = _lead_metrics(v_ego, radarstate)
  if lead is None: return False, (lead, d_rel, v_lead, closing, ttc, a_req)
  in_dist_window = (np.isfinite(d_rel) and (PREBRAKE_DIST_MIN_M <= d_rel <= dist_max))
  is_closing = closing > CLOSING_MIN_MPS
  slow_or_close = (v_lead <= lead_slow_mps) or (d_rel <= 18.0)
  ttc_trig = np.isfinite(ttc) and (ttc < ttc_start)
  req_trig = (a_req < a_req_start)
  trigger = bool(in_dist_window and is_closing and slow_or_close and (ttc_trig or req_trig))
  return trigger, (lead, d_rel, v_lead, closing, ttc, a_req)

def _prebrake_override(a_target: float, metrics):
  ttc_start, ttc_full, a_req_start, a_req_full, dist_max, lead_slow_mps, prebrake_max_decel = _derived_thresholds()
  lead, d_rel, v_lead, closing, ttc, a_req = metrics
  a_new = float(a_target)
  if np.isfinite(d_rel) and d_rel < HARD_MIN_LEAD_DIST_M:
    return float(min(a_new, prebrake_max_decel)), True
  if closing <= CLOSING_MIN_MPS: return a_new, False
  w_ttc = 0.0
  if np.isfinite(ttc) and ttc < ttc_start:
    w_ttc = (ttc_start - ttc) / max(ttc_start - ttc_full, 1e-3)
    w_ttc = float(np.clip(w_ttc, 0.0, 1.0))
  w_req = 0.0
  if a_req < a_req_start:
    w_req = (a_req_start - a_req) / max(a_req_start - a_req_full, 1e-3)
    w_req = float(np.clip(w_req, 0.0, 1.0))
  w_linear = float(max(w_ttc, w_req))
  if w_linear <= 0.0: return a_new, False
  curve_exponent = float(np.clip(2.0 - (closing / 10.0), 1.0, 2.0))
  w_smooth = w_linear ** curve_exponent 
  dynamic_max_decel = float(prebrake_max_decel)
  if w_linear > 0.8:
    panic_multiplier = 1.3 + max(0.0, (closing - 5.0) * 0.15)
    dynamic_max_decel = dynamic_max_decel * min(panic_multiplier, 2.5) 
  a_cmd = (1.0 - w_smooth) * a_new + w_smooth * dynamic_max_decel
  return float(min(a_new, a_cmd)), False

def _accel_clip_slew_step(dt: float, v_ego: float, lead_a: float, trigger_approach: bool, ttc: float, a_req: float) -> tuple[float, float]:
  base_slew_up = smooth_interp(v_ego, SLEW_V_BP, ACCEL_SLEW_RATE_BP)
  base_slew_down = smooth_interp(v_ego, SLEW_V_BP, DECEL_SLEW_RATE_BP)
  ttc_start, ttc_full, a_req_start, a_req_full, dist_max, lead_slow_mps, prebrake_max_decel = _derived_thresholds()
  ttc_fast = min(ttc_start + 0.4, 4.0)
  a_req_fast = min(a_req_start + 0.3, -0.2)
  fast_rel = bool(trigger_approach or (np.isfinite(ttc) and ttc < ttc_fast) or (a_req < a_req_fast))
  fast = (lead_a < ACCEL_CLIP_FAST_LEAD_DECEL_THRESH) or fast_rel
  if fast: base_slew_down = max(base_slew_down, 3.0)
  delta_up = float(max(0.01, base_slew_up * dt))
  delta_down = float(max(0.01, base_slew_down * dt))
  return delta_down, delta_up

class LongitudinalPlanner:
  def __init__(self, CP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    self.mpc.mode = 'acc'
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True
    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]
    self.output_a_target = 0.0
    self.output_should_stop = False
    
    # 時間記憶變數區
    self.smooth_coast_weight = 0.0
    self.clone_a_ema = 0.0  

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)
    self.solverExecutionTime = 0.0
    self.acm = ACM()
    self.aem = AEM()
    self.dtsc = DTSC(aggressiveness=0.8)

  @staticmethod
  def parse_model(model_msg):
    if (len(model_msg.position.x) == ModelConstants.IDX_N and len(model_msg.velocity.x) == ModelConstants.IDX_N and len(model_msg.acceleration.x) == ModelConstants.IDX_N):
      x = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.position.x)
      v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.velocity.x)
      a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.acceleration.x)
      j = np.zeros(len(T_IDXS_MPC))
    else:
      x, v, a, j = np.zeros(len(T_IDXS_MPC)), np.zeros(len(T_IDXS_MPC)), np.zeros(len(T_IDXS_MPC)), np.zeros(len(T_IDXS_MPC))
    throttle_prob = model_msg.meta.disengagePredictions.gasPressProbs[1] if len(model_msg.meta.disengagePredictions.gasPressProbs) > 1 else 1.0
    return x, v, a, j, throttle_prob

  def update(self, sm, dp_flags=0):
    mode = 'blended' if sm['selfdriveState'].experimentalMode else 'acc'

    if dp_flags & DPFlags.AEM:
      self.aem.update_states(model_msg=sm['modelV2'], radar_msg=sm['radarState'], v_ego=sm['carState'].vEgo)
      mode = self.aem.get_mode(mode)

    accel_coast = get_coast_accel(sm['carControl'].orientationNED[1]) if len(sm['carControl'].orientationNED) == 3 else ACCEL_MAX
    v_ego = sm['carState'].vEgo
    
    # 🌟 暴力切換：全域強制使用最乾淨的 ACC 傳統基底
    mode = 'acc'
    self.mpc.mode = mode

    lead_a = _get_lead_decel_a(sm)
    trigger_approach, metrics = _approach_trigger(v_ego, sm['radarState'])
    _lead, _d_rel, _v_lead, _closing, _ttc, _a_req = metrics
    has_lead = (_lead is not None and np.isfinite(_d_rel))

    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off
    force_slow_decel = sm['controlsState'].forceDecel
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    reset_state = reset_state or not v_cruise_initialized

    prev_accel_constraint = not (reset_state or sm['carState'].standstill)
    accel_clip = [ACCEL_MIN, get_max_accel(v_ego)]
    
    if mode == 'acc':
      steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['liveParameters'].angleOffsetDeg
      accel_clip = limit_accel_in_turns(v_ego, steer_angle_without_offset, accel_clip, self.CP)

    if has_lead and v_ego < 3.0 and _v_lead < 1.5 and _d_rel < 8.0:
      accel_clip[1] = min(accel_clip[1], max(0.2, _v_lead * 0.8))

    if reset_state:
      self.v_desired_filter.x = v_ego
      self.a_desired = np.clip(sm['carState'].aEgo, accel_clip[0], accel_clip[1])
      self.clone_a_ema = sm['carState'].aEgo 

    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    x, v, a, j, throttle_prob = self.parse_model(sm['modelV2'])
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    if not self.allow_throttle:
      clipped_accel_coast = max(accel_coast, accel_clip[0])
      clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED * 2], [accel_clip[1], clipped_accel_coast])
      accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    if force_slow_decel: v_cruise = 0.0

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)

    a_min_mpc, a_max_mpc = float(accel_clip[0]), float(accel_clip[1])
    if dp_flags & DPFlags.DTSC:
      lead_msg = sm['radarState'].leadOne if sm['radarState'].leadOne.status else None
      a_min_dtsc, a_max_dtsc = self.dtsc.get_mpc_constraints(sm['modelV2'], v_ego, accel_clip[0], accel_clip[1], v_set_mps=v_cruise, lead_msg=lead_msg)
      a_min_mpc = np.maximum(a_min_mpc, np.asarray(a_min_dtsc, dtype=float))
      a_max_mpc = np.minimum(a_max_mpc, np.asarray(a_max_dtsc, dtype=float))

    self.mpc.update(sm['radarState'], v_cruise, x, v, a, j, personality=sm['selfdriveState'].personality, a_min=a_min_mpc, a_max=a_max_mpc)
    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)

    if dp_flags & DPFlags.ACM:
      user_control = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
      self.acm.update_states(sm['carControl'], sm['radarState'], user_control, v_ego, v_cruise)
      self.a_desired_trajectory = self.acm.update_a_desired_trajectory(self.a_desired_trajectory)

    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw: cloudlog.info("FCW triggered")

    a_prev = self.a_desired
    self.a_desired = float(np.interp(self.dt, CONTROL_N_T_IDX, self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.a_desired + a_prev) / 2.0

    action_t = self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc, output_should_stop_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX, action_t=action_t, vEgoStopping=self.CP.vEgoStopping)
    
    mpc_a = float(output_a_target_mpc)
    e2e_a = float(sm['modelV2'].action.desiredAcceleration)

    if mode == 'acc':
      base_a_target = mpc_a
      self.output_should_stop = bool(output_should_stop_mpc)
    else:
      if has_lead:
        if mpc_a > 0.0 and e2e_a > -0.1: base_a_target = mpc_a
        else: base_a_target = min(mpc_a, e2e_a)
      else:
        e2e_is_stopping = bool(sm['modelV2'].action.shouldStop) or (e2e_a < -0.4)
        if e2e_is_stopping: base_a_target = min(mpc_a, e2e_a)
        elif v_ego < 3.0 and e2e_a > 0.0: base_a_target = min(mpc_a, e2e_a * 1.40)
        else: base_a_target = mpc_a
      self.output_should_stop = bool(sm['modelV2'].action.shouldStop) or bool(output_should_stop_mpc)

    final_a_target = base_a_target
    
    is_stopping_target = self.output_should_stop or (has_lead and _v_lead < 1.5 and _d_rel < 7.0)
    is_final_stop_zone = (not has_lead) or (has_lead and _d_rel < 4.0)

    # =========================================================================
    # 次世代：流水線狀態機 (老司機全取代版 - 速度導向優化)
    # =========================================================================

    # [狀態一] 緊急預煞 (防禦底線)
    if trigger_approach:
      final_a_target, hard_stop = _prebrake_override(base_a_target, metrics)
      if hard_stop: self.output_should_stop = True

    # [狀態二] 準備煞停區段 (確實煞停，移除防點頭)
    # =========================================================================
    elif is_stopping_target and v_ego < 10.0:
      if not is_final_stop_zone and base_a_target < 0.0:
        final_a_target = min(base_a_target, -0.15)
      
    # ==========================================
    # 🌟 核心革新：[狀態三] 🚦 塞車克隆模式 (Traffic Jam Clone)
    # ==========================================
    elif has_lead and (v_ego * CV.MS_TO_KPH < 35.0):
      w_clone = smooth_interp(v_ego * CV.MS_TO_KPH, [0.0, 35.0], [0.75, 0.25])
      
      # 1. 目標距離與非對稱死區
      target_dist = 5.0 + max(0.0, v_ego - 1.5) * 0.35
      dist_error = _d_rel - target_dist
      
      # 距離死區：允許拉遠 1.5 米，不允許逼近 (嚴格防禦)
      if 0.0 <= dist_error <= 1.5:
        dist_error_eff = 0.0
      elif dist_error > 1.5:
        dist_error_eff = dist_error - 1.5
      else:
        dist_error_eff = dist_error

      # 2. 🛡️ 智慧前饋衰減 (解決「遠處提早定竿」的元兇)
      if lead_a < 0.0:
        if _v_lead < 4.0:
          # 【滑行收尾模式】：前車快停了。如果我們還落後一大截 (dist_error > 4.0)，
          # 就放掉前車的重煞車訊號，不盲目照抄，把減速任務交給完美的滑行曲線！
          ff_weight = smooth_interp(dist_error, [1.0, 4.0], [1.0, 0.0])
        else:
          # 【動態跟車模式】：前車還在高速行進中減速，必須 100% 複製煞車保命
          ff_weight = 1.0
      else:
        # 前車加速：永遠 100% 跟上，防止起步遲鈍
        ff_weight = 1.0

      lead_a_feedforward = float(np.clip(lead_a, -2.0, 1.0)) * ff_weight

      # 3. 🚀 完美滑行曲線 (Glide Path)
      v_glide = dist_error_eff * 0.4
      ideal_v_ego = max(0.0, _v_lead + v_glide)
      v_error = ideal_v_ego - v_ego

      if v_error > 0.0:
        # 溫柔補油 (滑行/追擊)
        #v_comp = float(np.clip(v_error * 0.20, 0.0, 0.4))
        v_comp = float(np.clip(v_error * 0.35, 0.0, 1.0))
      else:
        # 動態煞車：距離越近，煞車越敏銳
        v_comp_factor = smooth_interp(_d_rel, [3.0, 12.0], [0.60, 0.20])
        v_comp = float(np.clip(v_error * v_comp_factor, -2.5, 0.0)) 
      
      raw_clone_a = lead_a_feedforward + v_comp

      # 4. 🛑 絕對駐車鎖死 (解決「停下後再蠕動」的元兇)
      # 當前車幾乎靜止，且我們已經滑進目標區塊附近 (dist_error < 2.0 米)
      if _v_lead < 0.5 and dist_error < 2.0:
        self.output_should_stop = True # 系統明確授權煞停
        if raw_clone_a > 0.0:
          raw_clone_a = 0.0 # 徹底沒收任何要求往前的推力
          
        # 當車速跌破 1.0 m/s，強迫給予深達 -0.8 的死區咬合力，克服變速箱怠速蠕動！
        brake_hold = smooth_interp(v_ego, [0.0, 1.0], [-0.50, -0.15])
        raw_clone_a = min(raw_clone_a, brake_hold)

      # 5. 非對稱微型濾波
      if _d_rel < 6.0 and lead_a < -0.5:
        self.clone_a_ema = raw_clone_a
      elif raw_clone_a < self.clone_a_ema:
        # 煞車方向：保持高度敏捷 (0.15 老 + 0.85 新)
        self.clone_a_ema = 0.15 * self.clone_a_ema + 0.85 * raw_clone_a
      else:
        # 加速/放煞車方向：慵懶濾波 (0.65 老 + 0.35 新)，消滅收油頓挫
        self.clone_a_ema = 0.3 * self.clone_a_ema + 0.7 * raw_clone_a
        
      final_a_target = (1.0 - w_clone) * base_a_target + w_clone * self.clone_a_ema
      self.smooth_coast_weight = 0.0

    # ==========================================
    # [狀態四] 🛣️ 高速巡航與熨斗 (High-Speed Pursuit & Iron)
    # ==========================================
    elif has_lead:
      w_speed_iron = smooth_interp(v_ego * CV.MS_TO_KPH, [65.0, 70.0], [0.7, 0.0])
      
      w_dist_iron = float(np.clip((_d_rel - 8.0) / 4.0, 0.0, 1.0))
      w_close_iron = float(np.clip((2.0 - abs(_closing)) / 1.5, 0.0, 1.0))
      raw_coast_weight = min(w_dist_iron, w_close_iron) * w_speed_iron

      if lead_a < -0.45 or lead_a > 0.5:
        raw_coast_weight = 0.0

      if raw_coast_weight > self.smooth_coast_weight:
        self.smooth_coast_weight += 0.02 * (raw_coast_weight - self.smooth_coast_weight)
      else:
        self.smooth_coast_weight += 0.15 * (raw_coast_weight - self.smooth_coast_weight)

      if self.smooth_coast_weight > 0.01:
        natural_coast = float(np.clip(accel_coast * 0.60 - 0.02, -0.25, 0.1))
        diff = final_a_target - natural_coast
        zone = 0.45
        if abs(diff) < zone:
          ratio = (abs(diff) / zone) ** 2
          smoothed_a = natural_coast + diff * ratio
        else:
          smoothed_a = final_a_target
        final_a_target = (1.0 - self.smooth_coast_weight) * final_a_target + self.smooth_coast_weight * smoothed_a
    
    # [狀態五] 💨 前車消失 / 淨空快速補油
    elif not has_lead and base_a_target > 0.0 and (v_ego * CV.MS_TO_KPH < 60.0) and v_ego < v_cruise - 1.0:
      clear_boost = smooth_interp(v_ego, [0.0, 5.0, 15.0], [0.30, 0.20, 0.0])
      if base_a_target < clear_boost:
        final_a_target = 0.5 * base_a_target + 0.5 * clear_boost
      self.smooth_coast_weight *= 0.6
      self.clone_a_ema = final_a_target
    else:
      self.smooth_coast_weight *= 0.6
      self.clone_a_ema = final_a_target

    # ==========================================
    # 收尾：Slew Rate 物理變化率限制
    # ==========================================
    delta_down, delta_up = _accel_clip_slew_step(self.dt, v_ego, lead_a, trigger_approach, _ttc, _a_req)
    for idx in range(2):
      accel_clip[idx] = np.clip(accel_clip[idx], self.prev_accel_clip[idx] - delta_down, self.prev_accel_clip[idx] + delta_up)

    self.output_a_target = float(np.clip(final_a_target, accel_clip[0], accel_clip[1]))
    self.prev_accel_clip = accel_clip

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')
    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'selfdriveState', 'radarState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()
    longitudinalPlan.hasLead = sm['radarState'].leadOne.status

    try:
      longitudinalPlan.longitudinalPlanSource = self.mpc.source
    except Exception:
      longitudinalPlan.longitudinalPlanSource = 'cruise'

    longitudinalPlan.fcw = self.fcw
    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)
