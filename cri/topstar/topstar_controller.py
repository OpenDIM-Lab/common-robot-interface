"""CRI trajectory and native servo adapter for the Topstar TRI100 / X5 SDK.

Positions are mm, angles are degrees, quaternions are (w, x, y, z).
The controller reference frame is X5 UF0 (world).
No vendor code is bundled here.
"""

import importlib
import math
import threading
import time
import warnings
from contextlib import contextmanager

import numpy as np
from transforms3d.euler import _AXES2TUPLE

from cri.controller import RobotController
from cri.transforms import euler2quat, quat2euler


class TopstarError(RuntimeError):
    """Base class for adapter errors."""


class TopstarSDKError(TopstarError):
    """SDK failure, retaining the original exception and vendor diagnostics."""

    def __init__(self, operation, cause):
        self.operation = operation
        self.original_error = cause
        self.error_code = getattr(cause, "error_code", None)
        self.error_symbol = getattr(cause, "error_symbol", None)
        self.error_message = getattr(cause, "error_message", str(cause))
        super().__init__("X5 {} failed (code={}): {}".format(
            operation, self.error_code, self.error_message))


class TopstarStateError(TopstarError):
    """The controller state does not permit the requested operation."""


class TopstarMotionError(TopstarError):
    """A submitted motion was cancelled, lost, or did not finish normally."""


class TopstarTargetError(TopstarMotionError, ValueError):
    """A target violates an IK or joint constraint before motion submission."""


class TopstarTimeoutError(TopstarMotionError, TimeoutError):
    """An adapter deadline expired; a stop is attempted for active motion."""


def _vector(value, size, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must contain {} finite numbers".format(name, size)) from exc
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError("{} must contain {} finite numbers".format(name, size))
    return result.copy()


def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError("{} must be positive and finite".format(name))
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("{} must be positive and finite".format(name))
    return value


def _integer(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("{} must be an integer in {}..{}".format(name, low, high))
    if not low <= value <= high:
        raise ValueError("{} must be in {}..{}".format(name, low, high))
    return int(value)


def _axes(value):
    if value not in _AXES2TUPLE:
        raise ValueError("euler_axes must be an explicit transforms3d Euler convention")
    return value


def _quaternion_pose(value):
    value = _vector(value, 7, "pose")
    norm = np.linalg.norm(value[3:])
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("pose quaternion must have nonzero finite norm")
    value[3:] /= norm
    return value


def x5_pose_to_quat(pose, euler_axes):
    """Convert X5 XYZABC (mm/deg) using an explicitly verified Euler order."""
    values = _vector([getattr(pose, k) for k in ("x", "y", "z", "a", "b", "c")],
                     6, "X5 pose")
    return euler2quat(values, _axes(euler_axes))


def quat_to_x5_pose(pose, euler_axes, sdk):
    """Convert a CRI quaternion pose to an SDK Pose (external axes default 0)."""
    values = quat2euler(_quaternion_pose(pose), _axes(euler_axes))
    return sdk.Pose(**dict(zip(("x", "y", "z", "a", "b", "c"), values)))


class TopstarController(RobotController):
    """Six-axis Topstar controller using ``import xapi.api as x5``.

    Construction only connects and reads. ``prepare_motion()`` explicitly
    selects AUTOCMD and enables servos, or the operator can prepare the cabinet.
    ``euler_axes`` has no implicit default: the supplied API manuals omit the
    compound rotation convention. command_speed sets the SDK per-command speed.
    command_acceleration and command_deceleration set separate percentages for
    trajectory moves. The cabinet manages its global override.

    ``tool_index=0`` uses the immutable flange frame and fits SyncRobot's
    software TCP. Indices 1..16 opt into a controller-managed TCP.
    """

    _REQUIRED_API = (
        "connect", "disconnect", "get_version", "get_all_system_state",
        "get_cjoint", "get_wpoint", "get_tf", "set_tf", "set_system_mode",
        "enable_servo", "movj", "movl", "movc",
        "get_cur_move_state", "wait_move_done", "wait_cmd_send_done", "abort", "stop",
        "cnvrt_p", "cnvrt_j", "cnvrt_p_to_p", "Pose", "Point", "Joint", "MovPointAdd",
    )

    def __init__(self, ip="192.168.1.6", *, sdk=None, euler_axes=None,
                 blend_radius=0, blend_mapping=None, tool_index=0,
                 move_timeout=60, state_timeout=5, poll_interval=0.02,
                 link_timeout=2, retry_num=5, joint_limits=None,
                 command_speed=1, command_acceleration=1, command_deceleration=1):
        self._ip = str(ip)
        self._sdk = sdk
        self._handle = None
        self._sdk_lock = threading.RLock()
        self._command_lock = threading.Lock()
        self._cancelled = threading.Event()
        self._stop_generation = 0
        self._moving = False
        self._servo_active = False
        self._servo_error = None
        self._servo_deadline = None
        self._servo_watchdog = None
        self._servo_watchdog_stop = threading.Event()
        self._commanded_joint_angles = None
        self._commanded_pose = None
        self._version = "not connected"
        self.last_motion_completed = None
        self.command_speed = command_speed
        self._command_acceleration = _integer(command_acceleration, 1, 100, "command_acceleration (%)")
        self._command_deceleration = _integer(command_deceleration, 1, 100, "command_deceleration (%)")
        self._euler_axes = None if euler_axes is None else _axes(euler_axes)
        self._tool_index = _integer(tool_index, 0, 16, "tool_index")
        self._move_timeout = _positive(move_timeout, "move_timeout")
        self._state_timeout = _positive(state_timeout, "state_timeout")
        self._poll_interval = _positive(poll_interval, "poll_interval")
        if self._poll_interval < 0.02:
            raise ValueError("poll_interval must be >= 0.02 seconds (X5 manual)")
        self._link_timeout = _integer(link_timeout, 1, 3600, "link_timeout")
        self._retry_num = _integer(retry_num, 0, 100, "retry_num")
        self._joint_limits = None
        if joint_limits is not None:
            limits = np.asarray(joint_limits, dtype=float)
            if (limits.shape != (6, 2) or not np.all(np.isfinite(limits)) or
                    np.any(limits[:, 0] >= limits[:, 1])):
                raise ValueError("joint_limits must contain six finite [min, max] degree pairs")
            self._joint_limits = limits.copy()
        self._blend_mapping = {}
        for radius, cnt in (blend_mapping or {}).items():
            radius = _positive(radius, "blend_mapping radius")
            self._blend_mapping[radius] = _integer(cnt, 0, 100, "cnt")
        self.blend_radius = blend_radius
        self.connect()

    def _call(self, operation, *args, **kwargs):
        with self._sdk_lock:
            if self._handle is None:
                raise TopstarStateError("X5 controller is disconnected")
            try:
                return getattr(self._sdk, operation)(self._handle, *args, **kwargs)
            except Exception as exc:
                raise TopstarSDKError(operation, exc) from exc

    @contextmanager
    def _exclusive(self):
        if not self._command_lock.acquire(blocking=False):
            raise TopstarStateError("Another controller command is active")
        try:
            yield
        finally:
            self._command_lock.release()

    def connect(self):
        """Connect or reconnect, without changing cabinet settings or enabling."""
        with self._exclusive(), self._sdk_lock:
            if self._handle is not None:
                return
            if self._sdk is None:
                try:
                    self._sdk = importlib.import_module("xapi.api")
                except (ImportError, OSError) as exc:
                    raise ImportError(
                        "Install the vendor X5 xapi wheel for CPython 3.10 and your platform") from exc
            missing = [name for name in self._REQUIRED_API if not hasattr(self._sdk, name)]
            if missing:
                raise TopstarStateError("X5 SDK lacks required APIs: " + ", ".join(missing))
            try:
                handle = self._sdk.connect(self._ip, link_timeout=self._link_timeout,
                                           retry_num=self._retry_num)
            except Exception as exc:
                raise TopstarSDKError("connect", exc) from exc
            if isinstance(handle, bool) or not isinstance(handle, int) or handle < 0:
                raise TopstarStateError("X5 connect returned an invalid handle: {!r}".format(handle))
            self._handle = handle  # Zero is a valid X5 handle.
            try:
                self._version = self._call("get_version")
                self._check_link(self.system_state)
                self._commanded_joint_angles = self.joint_angles
                self._commanded_pose = self.pose if self._euler_axes is not None else None
                self.last_motion_completed = None
                # Reconnection never clears a previous stop/failure latch.
            except BaseException as exc:
                try:
                    self._call("disconnect")
                except Exception as cleanup_exc:
                    exc.disconnect_error = cleanup_exc
                finally:
                    self._handle = None
                raise

    @property
    def connected(self):
        """Whether this adapter holds a handle; system_state.link checks the wire."""
        return self._handle is not None

    @property
    def tool_index(self):
        return self._tool_index

    @property
    def supports_software_tcp(self):
        """Avoid applying a software tool on top of an already selected TF."""
        return self._tool_index == 0

    @property
    def servo_active(self):
        return self._servo_active

    @property
    def servo_error(self):
        """Last asynchronous watchdog failure, cleared by prepare_motion()."""
        return self._servo_error

    @property
    def info(self):
        return "Topstar TRV007-710-S / TRI100; ip: {}; {}".format(self._ip, self._version)

    @property
    def system_state(self):
        return self._call("get_all_system_state")

    @property
    def system_alarm_info(self):
        """Read the controller's alarm queue on demand."""
        return self._call("get_system_alarm_info")

    @staticmethod
    def _check_link(state):
        if state.link != 1:
            raise TopstarStateError("X5 communication link is down")

    def _check_health(self, state, motion=False):
        self._check_link(state)
        if state.alarm != 0:
            error = TopstarStateError("X5 reports {} alarm(s); inspect the pendant".format(state.alarm))
            error.alarm_count = state.alarm
            raise error
        if state.rgm_state != 0:
            raise TopstarStateError("Exit X5 RGM mode before using CRI trajectory motion")
        if motion:
            if state.mode not in (3, 100) or state.enable != 1:
                raise TopstarStateError("Motion requires MANUAL(3) or AUTOCMD(100) and servo enable")
            if state.remote != 0:
                raise TopstarStateError("This adapter requires exclusive local control (remote=0)")
            if state.state not in (1, 3):
                raise TopstarStateError("X5 motion is paused or has an unknown run state")

    @staticmethod
    def _idle(state):
        return state.in_pos == 1 and state.cmd_queue == 0 and state.state == 3

    def _require_idle(self, motion=False):
        if self._servo_active:
            raise TopstarStateError("Stop the servo session before ordinary motion or configuration")
        state = self.system_state
        self._check_health(state, motion=motion)
        if not self._idle(state):
            raise TopstarStateError(
                "X5 must be stopped, axes in position, and command queue empty "
                f"(state={state.state}, in_pos={state.in_pos}, cmd_queue={state.cmd_queue}, "
                f"enable={state.enable}, mode={state.mode})"
            )
        return state

    def _require_axes(self):
        if self._euler_axes is None:
            raise TopstarStateError("Set euler_axes to the verified X5 XYZABC convention")
        return self._euler_axes

    def prepare_motion(self):
        """Explicitly select AUTOCMD and enable; never reset alarms/set remote.

        Also acknowledges a previous stop/failure and refreshes target state.
        The cabinet must already be stopped and in local control.
        """
        with self._exclusive():
            with self._sdk_lock:
                generation = self._stop_generation
            state = self._require_idle()
            if state.remote != 0:
                raise TopstarStateError("Servo preparation requires local control (remote=0)")

            def prepare_call(operation, *args):
                with self._sdk_lock:
                    if self._stop_generation != generation:
                        raise TopstarMotionError("Preparation interrupted by stop/abort/close")
                    self._call(operation, *args)

            try:
                if state.mode != 100:
                    prepare_call("set_system_mode", 100)
                    self._wait_state(lambda s: s.mode == 100, "AUTOCMD mode")
                if self.system_state.enable != 1:
                    prepare_call("enable_servo", True)
                    self._wait_state(lambda s: s.enable == 1, "servo enable")
                self._require_idle(motion=True)
                self._commanded_joint_angles = self.joint_angles
                self._commanded_pose = self.pose if self._euler_axes is not None else None
                with self._sdk_lock:
                    if self._stop_generation != generation:
                        raise TopstarMotionError("Preparation interrupted by stop/abort/close")
                    self._cancelled.clear()
                    self._servo_error = None
                self.last_motion_completed = None
            except BaseException:
                self._cancelled.set()
                raise

    def _wait_state(self, predicate, description):
        deadline = time.monotonic() + self._state_timeout
        while True:
            state = self.system_state
            self._check_health(state)
            if predicate(state):
                return
            if time.monotonic() >= deadline:
                raise TopstarTimeoutError("Timed out waiting for " + description)
            time.sleep(self._poll_interval)

    @property
    def tcp(self):
        return x5_pose_to_quat(self._call("get_tf", self._tool_index), self._require_axes())

    @tcp.setter
    def tcp(self, tcp):
        tcp = _quaternion_pose(tcp)
        with self._exclusive():
            self._require_idle()
            if self._tool_index == 0:
                if not (np.allclose(tcp[:3], 0, atol=1e-10) and
                        np.allclose(np.abs(tcp[3:]), (1, 0, 0, 0), atol=1e-10)):
                    raise ValueError("TF0 is the immutable flange; select tool_index=1..16 to set TCP")
                return
            state = self.system_state
            if state.mode not in (3, 100):
                raise TopstarStateError("set_tf requires MANUAL(3) or AUTOCMD(100)")
            self._commanded_pose = None  # Old Cartesian goal referred to a different TCP.
            target = quat_to_x5_pose(tcp, self._require_axes(), self._sdk)
            self._call("set_tf", self._tool_index, target)
            deadline = time.monotonic() + self._state_timeout
            while True:
                actual = self.tcp
                if (np.allclose(actual[:3], tcp[:3], atol=1e-6, rtol=0) and
                        abs(np.dot(actual[3:], tcp[3:])) >= 1 - 1e-10):
                    break
                if time.monotonic() >= deadline:
                    raise TopstarTimeoutError("Timed out confirming the selected TCP definition")
                time.sleep(self._poll_interval)

    @property
    def command_speed(self):
        """Per-command integer SDK percentage."""
        return self._command_speed

    @command_speed.setter
    def command_speed(self, value):
        self._command_speed = _integer(value, 1, 100, "command_speed (%)")

    @property
    def linear_speed(self):
        raise NotImplementedError("Topstar uses command_speed (%)")

    @linear_speed.setter
    def linear_speed(self, speed):
        raise NotImplementedError("Topstar uses command_speed (%)")

    @property
    def angular_speed(self):
        raise NotImplementedError("Topstar uses command_speed (%)")

    @angular_speed.setter
    def angular_speed(self, speed):
        raise NotImplementedError("Topstar uses command_speed (%)")

    @property
    def blend_radius(self):
        return self._blend_radius

    @blend_radius.setter
    def blend_radius(self, radius):
        radius = float(radius)
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("blend_radius must be nonnegative and finite (mm)")
        if self._servo_active and radius != 0:
            raise ValueError("Native servo targets require blend_radius=0")
        if radius and radius not in self._blend_mapping:
            raise ValueError("Nonzero blend_radius needs an explicit calibrated blend_mapping entry")
        self._blend_radius = radius

    def _additional(self, *, wrist=0):
        cnt = -1 if self.blend_radius == 0 else self._blend_mapping[self.blend_radius]
        return self._sdk.MovPointAdd(vel=self.command_speed, cnt=cnt,
                                    acc=self._command_acceleration, dec=self._command_deceleration,
                                    offset=-2, toffset=-2, vision_num=-1,
                                    track_num=-1, wrist=wrist, inc=0, search=0, rtcp=0)

    def _joint_values(self, joint):
        return _vector([getattr(joint, "j{}".format(i)) for i in range(1, 7)], 6, "X5 joints")

    def _check_joints(self, values):
        values = _vector(values, 6, "joint_angles")
        if self._joint_limits is not None:
            low, high = self._joint_limits.T
            outside = np.flatnonzero((values < low) | (values > high))
            if outside.size:
                detail = ", ".join(f"J{i + 1}={values[i]:.3f} deg (allowed {low[i]:g}..{high[i]:g})"
                                   for i in outside)
                raise TopstarTargetError(f"Joint target exceeds configured joint_limits: {detail}")
        return values

    @property
    def joint_angles(self):
        return self._joint_values(self._call("get_cjoint"))

    @property
    def commanded_joint_angles(self):
        if self._commanded_joint_angles is None:
            raise TopstarStateError("Joint target is unknown; call prepare_motion() to resynchronize")
        return self._commanded_joint_angles.copy()

    def _world_point(self):
        point = self._call("get_wpoint")
        if point.uf != 0 or point.tf != self._tool_index:
            point = self._call("cnvrt_p_to_p", point, 0, self._tool_index)
        if point.uf != 0 or point.tf != self._tool_index:
            raise TopstarStateError("X5 coordinate conversion returned an inconsistent UF/TF")
        return point

    @property
    def pose(self):
        axes = self._require_axes()
        return x5_pose_to_quat(self._world_point().pose, axes)

    @property
    def commanded_pose(self):
        if self._commanded_pose is None:
            raise TopstarStateError("Cartesian target is unknown; configure axes and call prepare_motion()")
        return self._commanded_pose.copy()

    @property
    def elbow(self):
        return None

    @property
    def commanded_elbow(self):
        return None

    def _target_point(self, pose, current):
        target_pose = quat_to_x5_pose(pose, self._require_axes(), self._sdk)
        for key in ("e1", "e2", "e3"):
            setattr(target_pose, key, getattr(current.pose, key))
        # Preserve current arm branch and J1/J4/J6 turns; do not invent cfg.
        return self._sdk.Point(pose=target_pose, uf=0, tf=self._tool_index,
                               cfg=tuple(current.cfg))

    def _resolve_linear_target(self, pose, current, seed):
        target = self._target_point(pose, current)
        locked = self._call("cnvrt_j", target, 1, seed)
        values = self._joint_values(locked)
        continuous = values.copy()
        # Conf&Turn keeps the seed's turns even across a ±180-degree boundary.
        # Choose the equivalent angle nearest the seed without changing arm shape.
        axes = [0, 3, 5]
        current_joints = self._joint_values(seed)
        continuous[axes] += 360 * np.rint((current_joints[axes] - values[axes]) / 360)
        continuous = self._check_joints(continuous)
        if np.array_equal(continuous, values):
            return target, continuous

        data = {f"j{i + 1}": value for i, value in enumerate(continuous)}
        data.update({key: getattr(locked, key) for key in ("e1", "e2", "e3")})
        verified = self._call("cnvrt_p", self._sdk.Joint(**data), 0, self._tool_index)
        verified_pose = x5_pose_to_quat(verified.pose, self._require_axes())
        if (verified.uf != target.uf or verified.tf != target.tf or
                verified.cfg[3] != target.cfg[3] or
                np.linalg.norm(verified_pose[:3] - pose[:3]) > 1e-3 or
                abs(np.dot(verified_pose[3:], pose[3:])) < 1 - 1e-9):
            raise TopstarMotionError("X5 equivalent turn did not preserve the requested pose and arm configuration")
        target = self._sdk.Point(pose=target.pose, uf=target.uf, tf=target.tf, cfg=tuple(verified.cfg))
        confirmed = self._joint_values(self._call("cnvrt_j", target, 1, seed))
        if not np.allclose(confirmed, continuous, atol=1e-4, rtol=0):
            raise TopstarMotionError("X5 could not confirm the equivalent turn for the linear target")
        return target, self._check_joints(confirmed)

    @contextmanager
    def linear_path_checker(self):
        """Predict an ordered linear path from the current stationary joints."""
        with self._exclusive():
            self._require_idle()
            current = self._world_point()
            seed = self._call("get_cjoint")
            initial = self._joint_values(seed)

            def check(poses):
                nonlocal current, seed
                point, joint = current, seed
                for index, pose in enumerate(poses, 1):
                    point, joint = self._preview_linear_target(pose, point, joint, index)
                current, seed = point, joint

            yield check
            self._require_idle()
            if not np.allclose(self.joint_angles, initial, atol=0.05, rtol=0):
                raise TopstarStateError("Robot position changed during the linear path check")

    def _preview_linear_target(self, pose, point, seed, label):
        try:
            point, joints = self._resolve_linear_target(_quaternion_pose(pose), point, seed)
        except TopstarTargetError as exc:
            raise TopstarTargetError(f"Linear preflight target {label}: {exc}") from exc
        except TopstarSDKError as exc:
            # Only vendor IK and axis-limit errors can reject an individual candidate.
            if exc.operation != "cnvrt_j" or exc.error_code not in (-120, -1021):
                raise
            self._require_idle()
            raise TopstarTargetError(f"Linear preflight target {label}: {exc}") from exc
        return point, self._sdk.Joint(*joints, point.pose.e1, point.pose.e2, point.pose.e3)

    def check_linear_targets(self, poses):
        """Check consecutive target inverse solutions without submitting motion."""
        with self.linear_path_checker() as check:
            check(poses)

    def _motion(self, build):
        with self._exclusive():
            if self._cancelled.is_set():
                raise TopstarStateError("Motion is latched off; inspect the robot and call prepare_motion()")
            self._require_axes()
            self._require_idle(motion=True)
            operation, args, kwargs, joints, pose = build()
            try:
                with self._sdk_lock:
                    if self._cancelled.is_set():
                        raise TopstarMotionError("Motion cancelled before submission")
                    # Recheck immediately before sending; IK can take time.
                    self._require_idle(motion=True)
                    self._moving = True  # A failed SDK call may have partially sent a command.
                    self.last_motion_completed = False
                    self._call(operation, *args, **kwargs)
                    self._commanded_joint_angles = joints.copy()
                    self._commanded_pose = pose.copy()
                self._wait_motion()
                self.last_motion_completed = True
            except BaseException as exc:
                self._commanded_joint_angles = None
                self._commanded_pose = None
                try:
                    self._halt("abort")
                except Exception as stop_exc:
                    # Preserve the initiating failure and expose stop uncertainty.
                    exc.stop_error = stop_exc
                    warnings.warn("X5 abort/stop confirmation failed: {}".format(stop_exc), RuntimeWarning)
                if getattr(exc, "alarm_count", 0):
                    # X5 reads the alarm queue from a database. Query once after abort.
                    try:
                        details = self.system_alarm_info
                    except TopstarError as alarm_exc:
                        exc.args = (f"{exc}\nCould not read X5 alarm details: {alarm_exc}",)
                    else:
                        exc.args = (f"{exc}\nX5 alarm details: {details!r}",)
                raise
            finally:
                with self._sdk_lock:
                    self._moving = False

    def move_joints(self, joint_angles):
        values = self._check_joints(joint_angles)

        def build():
            additional = self._additional()
            current = self._call("get_cjoint")
            data = {"j{}".format(i + 1): value for i, value in enumerate(values)}
            data.update({k: getattr(current, k) for k in ("e1", "e2", "e3")})
            target = self._sdk.Joint(**data)
            point = self._call("cnvrt_p", target, 0, self._tool_index)
            pose = x5_pose_to_quat(point.pose, self._require_axes())
            return "movj", (target,), {"additional_data": additional}, values, pose

        self._motion(build)

    def move_linear(self, pose, elbow=None, *, wrist=0):
        if elbow is not None:
            raise ValueError("TRV007-710-S is a six-axis arm; elbow is unsupported")
        pose = _quaternion_pose(pose)
        wrist = _integer(wrist, 0, 1, "wrist")

        def build():
            additional = self._additional(wrist=wrist)
            target, joints = self._resolve_linear_target(pose, self._world_point(), self._call("get_cjoint"))
            return "movl", (target,), {"additional_data": additional}, joints, pose

        self._motion(build)

    def move_circular(self, via_pose, end_pose, elbow=None):
        if elbow is not None:
            raise ValueError("TRV007-710-S is a six-axis arm; elbow is unsupported")
        via_pose, end_pose = _quaternion_pose(via_pose), _quaternion_pose(end_pose)

        def build():
            additional = self._additional()
            current = self._world_point()
            start = x5_pose_to_quat(current.pose, self._require_axes())[:3]
            first, second = via_pose[:3] - start, end_pose[:3] - start
            scale = np.linalg.norm(first) * np.linalg.norm(second)
            if scale < 1e-12 or np.linalg.norm(np.cross(first, second)) <= 1e-9 * scale:
                raise ValueError("Circular start, via and end positions must be distinct and noncollinear")
            via, target = self._target_point(via_pose, current), self._target_point(end_pose, current)
            via_joint = self._call("cnvrt_j", via, 1, self._call("get_cjoint"))
            self._check_joints(self._joint_values(via_joint))
            joints = self._check_joints(self._joint_values(self._call("cnvrt_j", target, 1, via_joint)))
            return ("movc", (via, target),
                    {"central_angle": 0, "additional_data": additional}, joints, end_pose)

        self._motion(build)

    def prepare_servo(self, *, period=0.032, gain=5, acceleration=10,
                      watchdog_timeout=0.5):
        """Begin native SDK servol/servoj submissions after prepare_motion().

        Each servo call acknowledges submission, not arrival. This uses the
        documented non-RGM SDK APIs; it does not provide hard realtime timing.
        A gap after the first target triggers abort and latches motion off.
        """
        period = _positive(period, "servo period (s)")
        if not 0.002 <= period <= 2:
            raise ValueError("servo period must be in 0.002..2 seconds")
        gain = _integer(gain, 0, 50, "servo gain")
        acceleration = _integer(acceleration, 1, 100, "servo acceleration (%)")
        watchdog_timeout = _positive(watchdog_timeout, "servo watchdog_timeout (s)")
        if watchdog_timeout <= max(period, self._poll_interval):
            raise ValueError("watchdog_timeout must exceed period and poll_interval")
        with self._exclusive(), self._sdk_lock:
            if self._cancelled.is_set():
                raise TopstarStateError("Call prepare_motion() before preparing servo submissions")
            for operation in ("servol", "servoj"):
                if not callable(getattr(self._sdk, operation, None)):
                    raise NotImplementedError("X5 SDK does not provide " + operation)
            self._require_axes()
            self._require_idle(motion=True)
            if self.blend_radius != 0:
                raise ValueError("Native servo targets require blend_radius=0")
            self._servo_period = period
            self._servo_gain = gain
            self._servo_acceleration = acceleration
            self._servo_timeout = watchdog_timeout
            self._servo_deadline = None
            self._servo_error = None
            self._servo_active = True
            event = threading.Event()
            self._servo_watchdog_stop = event
            worker = threading.Thread(target=self._watch_servo, args=(event,), daemon=True)
            self._servo_watchdog = worker
            try:
                worker.start()
            except BaseException:
                self._servo_active = False
                event.set()
                raise

    def _watch_servo(self, event):
        while not event.wait(self._poll_interval):
            with self._sdk_lock:
                if event is not self._servo_watchdog_stop or not self._servo_active:
                    return
                if self._servo_deadline is None or time.monotonic() < self._servo_deadline:
                    continue
                error = TopstarTimeoutError("Servo targets stopped arriving; aborting X5 motion")
                self._servo_error = error
                self._cancelled.set()
                self._commanded_joint_angles = self._commanded_pose = None
                # Keep the lock through abort initiation so close/reconnect
                # cannot make this watchdog act on a different connection.
                try:
                    self._halt("abort")
                except Exception as stop_error:
                    error.stop_error = stop_error
                    warnings.warn("X5 servo watchdog stop failed: {}".format(stop_error), RuntimeWarning)
                return

    def _servo_motion(self, build):
        with self._exclusive():
            with self._sdk_lock:
                if self._servo_error is not None:
                    raise self._servo_error
                if self._cancelled.is_set() or not self._servo_active:
                    raise TopstarStateError("Call prepare_motion() and prepare_servo() before servo targets")
            try:
                with self._sdk_lock:
                    self._check_servo_deadline()
                self._check_health(self.system_state, motion=True)
                operation, target, joints, pose, percent = build()
                with self._sdk_lock:
                    if self._cancelled.is_set() or not self._servo_active:
                        raise TopstarMotionError("Servo target cancelled before submission")
                    self._check_health(self.system_state, motion=True)
                    self._check_servo_deadline()
                    self._moving = True
                    self.last_motion_completed = False
                    self._call(operation, target, cmdt=self._servo_period, lookahead=0,
                               gain=self._servo_gain, vel=percent, acc=self._servo_acceleration)
                    self._commanded_joint_angles = joints.copy()
                    self._commanded_pose = pose.copy()
                    self._servo_deadline = time.monotonic() + self._servo_timeout
            except BaseException as error:
                self._commanded_joint_angles = self._commanded_pose = None
                try:
                    self._halt("abort")
                except Exception as stop_error:
                    error.stop_error = stop_error
                    warnings.warn("X5 servo abort failed: {}".format(stop_error), RuntimeWarning)
                raise

    def _check_servo_deadline(self):
        # Also check on the submitting thread: a late target must not revive
        # an expired stream merely because the watchdog has not been scheduled.
        if self._servo_deadline is not None and time.monotonic() >= self._servo_deadline:
            error = TopstarTimeoutError("Servo target arrived after watchdog_timeout")
            self._servo_error = error
            raise error

    def servo_move_linear(self, pose, elbow=None):
        """Submit one UF0/tool_index target to native SDK servol, without waiting."""
        if elbow is not None:
            raise ValueError("TRV007-710-S is a six-axis arm; elbow is unsupported")
        pose = _quaternion_pose(pose)

        def build():
            percent = self.command_speed
            target = self._target_point(pose, self._world_point())
            joints = self._check_joints(self._joint_values(
                self._call("cnvrt_j", target, 1, self._call("get_cjoint"))))
            return "servol", target, joints, pose, percent

        self._servo_motion(build)

    def servo_move_joints(self, joint_angles):
        """Submit one six-axis target to native SDK servoj, without waiting."""
        values = self._check_joints(joint_angles)

        def build():
            percent = self.command_speed
            current = self._call("get_cjoint")
            data = {"j{}".format(i + 1): value for i, value in enumerate(values)}
            data.update({key: getattr(current, key) for key in ("e1", "e2", "e3")})
            target = self._sdk.Joint(**data)
            point = self._call("cnvrt_p", target, 0, self._tool_index)
            pose = x5_pose_to_quat(point.pose, self._require_axes())
            return "servoj", target, values, pose, percent

        self._servo_motion(build)

    def _wait_motion(self):
        deadline = time.monotonic() + self._move_timeout
        while True:
            if self._cancelled.is_set():
                raise TopstarMotionError("Motion cancelled by stop/abort/close")
            if time.monotonic() >= deadline:
                raise TopstarTimeoutError("X5 motion timed out after {} s".format(self._move_timeout))
            state = self.system_state
            self._check_health(state, motion=True)
            status = self._call("get_cur_move_state")
            if status == 2:
                raise TopstarMotionError("X5 motion was not found (queue cleared or command lost)")
            if status not in (0, 1):
                raise TopstarMotionError("Unknown X5 motion state: {!r}".format(status))
            if time.monotonic() >= deadline:
                raise TopstarTimeoutError("X5 motion timed out after {} s".format(self._move_timeout))
            if status == 1 and state.in_pos == 1 and state.cmd_queue == 0:
                remaining_ms = max(1, math.ceil((deadline - time.monotonic()) * 1000))
                # xapi 3.2.5 waits 20 ms before its first completion check,
                # then polls at up to 50 ms intervals. A 20 ms timeout only
                # checks the SDK's additional completion flags once.
                try:
                    self._call("wait_move_done", timeout_ms=min(remaining_ms, 100))
                except TopstarSDKError as exc:
                    if exc.error_code != -98:  # XERR_TIMEOUT
                        raise
                    # Poll the same move again within the overall deadline.
                else:
                    final = self.system_state
                    self._check_health(final, motion=True)
                    if (self._call("get_cur_move_state") != 1 or
                            final.in_pos != 1 or final.cmd_queue != 0 or self._cancelled.is_set()):
                        raise TopstarMotionError("X5 completion was invalidated before confirmation")
                    if time.monotonic() >= deadline:
                        raise TopstarTimeoutError("X5 completion exceeded move_timeout")
                    return
            time.sleep(min(self._poll_interval, max(0.0, deadline - time.monotonic())))

    def _halt(self, method):
        with self._sdk_lock:
            was_servo = self._servo_active or self._servo_deadline is not None
            self._servo_active = False
            self._servo_deadline = None
            self._servo_watchdog_stop.set()
            self._stop_generation += 1
            self._cancelled.set()  # Atomic with command submission.
            self._call(method)
            # A cached idle snapshot alone cannot prove a queued stop was sent.
            self._call("wait_cmd_send_done")
        deadline = time.monotonic() + self._state_timeout
        while True:
            state = self.system_state
            self._check_link(state)
            if state.in_pos == 1 and state.cmd_queue == 0:
                if was_servo:
                    self._moving = False
                return
            if time.monotonic() >= deadline:
                raise TopstarTimeoutError("X5 {} sent, but axes/queue did not confirm stopped".format(method))
            time.sleep(self._poll_interval)

    def stop(self):
        """Decelerate/clear queued motion, latch further submissions off."""
        self._halt("stop")

    def abort(self):
        """Stop all cabinet actions and latch further submissions off."""
        self._halt("abort")

    def disable_servo(self):
        """Confirm stop and servo-off after a blocking command has returned.

        Further motion stays latched off until prepare_motion() is called.
        This is explicit; close() retains its existing disconnect-only policy.
        """
        from cri.topstar.sdk_session import disable_servo

        with self._exclusive(), self._sdk_lock:
            if self._handle is None:
                raise TopstarStateError("X5 controller is disconnected")
            self._stop_generation += 1
            self._cancelled.set()
            self._servo_active = False
            self._servo_deadline = None
            self._servo_watchdog_stop.set()
            try:
                disable_servo(self._sdk, self._handle, force_stop=self._moving,
                              timeout=self._state_timeout)
                self._moving = False
            except Exception as error:
                raise TopstarSDKError("disable_servo", error) from error

    def close(self):
        """Stop this adapter's active motion and disconnect. Idempotent.

        Does not disable servos, reset alarms, restore TF slots, or shut down
        the SDK's process-global logger. Stop failures are reported after
        disconnect has still been attempted.
        """
        error = None
        with self._sdk_lock:
            if self._handle is None:
                return
            self._stop_generation += 1
            self._cancelled.set()
            self._servo_active = False
            self._servo_watchdog_stop.set()
        try:
            if self._moving:
                self._halt("abort")
        except Exception as exc:
            error = exc
        finally:
            with self._sdk_lock:
                try:
                    if self._handle is not None:
                        self._call("disconnect")
                except Exception as exc:
                    if error is None:
                        error = exc
                finally:
                    self._handle = None
        if error is not None:
            raise error

    disconnect = close
