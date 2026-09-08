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

A_CRUISE_MAX_VALS = [1.25,  1.15,   1.05,   0.7, 0.644, 0.441, 0.198] 
A_CRUISE_MAX_BP   = [0.0,  2.78,  8.33,  15.0,  20.0,  25.0,  30.0]

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

    if v_ego * CV.MS_TO_KPH >= 25.0:
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
      self.clone_a_ema = sm['carState'].aEgo # 重置克隆狀態

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
    
    # 1. 先讀取原始 MPC 的碰撞警告
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    
    # ==========================================================
    # 🌟 配套修復：接管碰撞警告 (FCW) 判斷權
    # ==========================================================
    # 底層 MPC 非常保守，它不知道我們在使用「克隆模式」，看到我們跟腳貼近就會亂叫。
    # 所以在 35 km/h 以下，我們將 FCW 的發言權搶過來，交給我們的 [狀態一] 緊急預煞模組！
    if has_lead and (v_ego * CV.MS_TO_KPH < 35.0):
      # 只有當雷達物理數值真的突破底線，觸發我們的緊急預煞時，才允許發出警報聲！
      self.fcw = bool(trigger_approach)

    if self.fcw: cloudlog.info("FCW triggered")

    a_prev = self.a_desired
    self.a_desired = float(np.interp(self.dt, CONTROL_N_T_IDX, self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.a_desired + a_prev) / 2.0

    action_t = self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc, output_should_stop_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX, action_t=action_t, vEgoStopping=self.CP.vEgoStopping)
    
    mpc_a = float(output_a_target_mpc)
    e2e_a = float(sm['modelV2'].action.desiredAcceleration)

    # ==========================================================
    # 🌟 全局基底訊號淨化區 (打破 acc/blended 模式限制)
    # ==========================================================
    
    # 1. 預設基底：100% 信任平穩的傳統 MPC
    # 無論在什麼模式，只要不符合特殊條件，一律用最平滑的 mpc_a 當作基底
    base_a_target = mpc_a
    self.output_should_stop = bool(output_should_stop_mpc)

    # 2. 解除封印：全域紅綠燈與路口判斷 (跨越 25km/h 限制)
    # 條件：前方無車 (有車時一律優先跟車保命) 且 時速低於 70 km/h
    if not has_lead and (v_ego * CV.MS_TO_KPH < 70.0):
      # 直接讀取視覺神經網路的煞停意圖
      e2e_is_stopping = bool(sm['modelV2'].action.shouldStop) or (e2e_a < -0.4)
      
      if e2e_is_stopping:
        # 🚨 強制介入：只要視覺模型看到紅綠燈或停止線，立刻取最保守的煞車力道
        base_a_target = min(mpc_a, e2e_a)
        # 點亮煞停旗標，確保後續的 Planner 狀態機能順利進入煞停程序
        self.output_should_stop = True

    final_a_target = base_a_target
    
    is_stopping_target = self.output_should_stop or (has_lead and _v_lead < 0.5 and _d_rel < 15.0)
    is_final_stop_zone = (not has_lead) or (has_lead and _d_rel < 7.0)

    # =========================================================================
    # 次世代：流水線狀態機 (刪除干擾點段差，100%交由克隆模式)
    # =========================================================================

    # [狀態一] 緊急預煞 (防禦底線)
    # 🌟 配套修復 2：嚴格限制預煞介入條件，防止它在塞車時搶奪克隆模式的控制權
    # 條件A (塞車極端防護)：35km/h 以下，除非速差超過 2.0 m/s 且 距離小於 3 米，才准介入！
    is_panic_jam = has_lead and (v_ego * CV.MS_TO_KPH < 35.0) and (_closing > 2.0) and (_d_rel < 3.0)
    
    # 條件B (高速常規防護)：35km/h 以上，維持原本的 trigger_approach 邏輯
    is_high_speed_approach = (v_ego * CV.MS_TO_KPH >= 35.0) and trigger_approach

    if is_panic_jam or is_high_speed_approach:
      final_a_target, hard_stop = _prebrake_override(base_a_target, metrics)
      if hard_stop: self.output_should_stop = True
      
      # 只有在真正命懸一線、被強制接管煞車時，才觸發警報！
      self.fcw = True

    # 🌟 移除獨立的狀態二，避免硬核覆蓋干擾克隆模式，將「防點頭」移至全域末端

    
    # ==========================================
    # [狀態三] 🚦 塞車克隆模式 (Traffic Jam Clone)
    # ==========================================
    # 完美涵蓋 0-35 km/h，包括自然滑順的動態跟車
    elif has_lead and (v_ego * CV.MS_TO_KPH < 35.0):
      # 基礎克隆權重
      w_clone = smooth_interp(v_ego * CV.MS_TO_KPH, [30.0, 35.0], [1.0, 0.0])
      
      # =========================================================
      # 🌟 核心修復：【駐車交接機制 (Parking Handover)】
      # 解決純 PD 控制沒有「踩死煞車」概念導致的點放煞車與無限蠕動。
      # 當前車即將靜止且距離拉近時，將控制權無縫交還給擅長煞停的 MPC。
      # =========================================================
      
      # 1. 判斷前車是否靜止 (低於 2.0m/s 開始準備，低於 0.5m/s 視為完全靜止)
      w_lead_stopped = smooth_interp(_v_lead, [0.5, 2.0], [1.0, 0.0])
      
      # 2. 判斷距離多近 (距離 10m 開始淡出克隆，距離 5m 時徹底交還給 MPC)
      w_dist_yield = smooth_interp(_d_rel, [6.0, 15.0], [0.0, 1.0])
      
      # 3. 權重融合：前車若沒停，不影響克隆；前車若停了，距離越近克隆權重越低
      final_yield_factor = 1.0 - (w_lead_stopped * (1.0 - w_dist_yield))
      w_clone = min(w_clone, final_yield_factor)
      
      # =========================================================

      # 1. 基礎前饋 (Feedforward)
      lead_a_feedforward = float(np.clip(lead_a, -2.0, 1.0))
    
      # =========================================================
      # 🌟 魔法擴充：動態解開封印 (Dynamic Gap Closing)
      # 解決前車消失時，面對大空檔加速遲鈍的問題。
      # 當誤差大於 2 公尺時，開始漸漸放寬油門上限與濾波限制！
      # =========================================================
      target_dist = 2.0 + v_ego * 1.0
      dist_error = _d_rel - target_dist
      
      # 距離誤差越大，允許的最大加速力道就越高 (從 0.2 解放至 0.8)
      p_max = smooth_interp(dist_error, [2.0, 8.0], [0.2, 0.8])
      v_max = smooth_interp(dist_error, [2.0, 8.0], [0.3, 0.8])
      
      # 距離誤差越大，加速時的濾波器就越弱，讓起步更果斷 (從 0.7 降至 0.2)
      up_filter_weight = smooth_interp(dist_error, [2.0, 8.0], [0.7, 0.2])

      # 2. 距離補償 (P)
      if 0.0 < dist_error < 1.5:
        p_comp = 0.0
      elif -0.5 < dist_error <= 0.0:
        p_comp = 0.0
      else:
        comp_factor = 0.03 if dist_error > 0 else 0.08
        # 🌟 套用動態油門上限 p_max
        p_comp = float(np.clip(dist_error * comp_factor, -0.6, p_max)) 
      
      # 3. 速差補償 (D)
      v_error = -_closing
      
      if 0.0 < v_error < 0.5:
        v_comp = 0.0
      else:
        v_comp_factor = 0.10 if v_error > 0 else 0.25
        # 🌟 套用動態油門上限 v_max
        v_comp = float(np.clip(v_error * v_comp_factor, -1.2, v_max)) 
      
      raw_clone_a = lead_a_feedforward + p_comp + v_comp

      # 4. 微型濾波
      if raw_clone_a < self.clone_a_ema:
        # 收油/煞車：維持瞬間反應
        self.clone_a_ema = 0.1 * self.clone_a_ema + 0.9 * raw_clone_a
      else:
        # 加速/補油：🌟 套用動態濾波權重 up_filter_weight
        self.clone_a_ema = up_filter_weight * self.clone_a_ema + (1.0 - up_filter_weight) * raw_clone_a
      
      # 保命底線
      if _d_rel < 6.0 and lead_a < -0.5:
        self.clone_a_ema = raw_clone_a
        
      # 5. 與底層 MPC 完美融合
      final_a_target = (1.0 - w_clone) * base_a_target + w_clone * self.clone_a_ema
      #final_a_target = 0.4 * base_a_target + 0.6 * self.clone_a_ema
      
      self.smooth_coast_weight = 0.0

    # ==========================================
    # [狀態四] 🛣️ 高速巡航與熨斗 (High-Speed Pursuit & Iron)
    # ==========================================
    elif has_lead:
      w_speed_iron = smooth_interp(v_ego * CV.MS_TO_KPH, [65.0, 70.0], [1.0, 0.0])
      w_dist_iron = float(np.clip((_d_rel - 8.0) / 4.0, 0.0, 1.0))
      w_close_iron = float(np.clip((2.0 - abs(_closing)) / 1.5, 0.0, 1.0))
      raw_coast_weight = min(w_dist_iron, w_close_iron) * w_speed_iron

      if lead_a < -0.6 or lead_a > 0.8:
        raw_coast_weight = 0.0

      if raw_coast_weight > self.smooth_coast_weight:
        self.smooth_coast_weight += 0.02 * (raw_coast_weight - self.smooth_coast_weight)
      else:
        self.smooth_coast_weight += 0.25 * (raw_coast_weight - self.smooth_coast_weight)

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
    # 🌟 全域防點頭機制 (Global Anti-Nod)
    # ==========================================
    # 無論是克隆模式還是無車煞停，在最後 1.5 m/s 且正在煞車時，優雅地微放煞車
    if is_stopping_target and is_final_stop_zone and v_ego < 1.5 and final_a_target < -0.2:
      nod_relief = (1.5 - v_ego) / 1.5 * 0.40
      final_a_target = min(final_a_target + nod_relief, -0.15)

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
