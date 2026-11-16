"""Robot class provide a common, higher-level interface to various robot arms.
"""

from abc import ABC, abstractmethod
from concurrent.futures import Future
from contextlib import contextmanager
from threading import Lock, Thread

import numpy as np

from cri.transforms import euler2quat, quat2euler, transform, inv_transform, frame


class InvalidEulerAxes(ValueError):
    pass


class InvalidJointAngles(ValueError):
    pass


class InvalidEulerPose(ValueError):
    pass


class AsyncBusy(RuntimeError):
    pass


class AsyncNotBusy(RuntimeError):
    pass    


def check_joint_angles(joint_angles):
    if not (4 <= len(joint_angles) <= 7):
        raise InvalidJointAngles


def check_pose(pose):
    if len(pose) != 6:
        raise InvalidEulerPose


class Robot(ABC):
    """Robot class provides common robot interface.

    Poses and coordinate frames are specified using 3D Cartesian positions
    and Euler rotations.  This makes it easier to specify poses and coordinate
    frames using a more intuitive representation than quaternions.
    """

    EULER_AXES = ('sxyz', 'sxyx', 'sxzy', 'sxzx', 'syzx', 'syzy',
                  'syxz', 'syxy', 'szxy', 'szxz', 'szyx', 'szyz',
                  'rzyx', 'rxyx', 'ryzx', 'rxzx', 'rxzy', 'ryzy',
                  'rzxy', 'ryxy', 'ryxz', 'rzxz', 'rxyz', 'rzyz',
                  )

    def __repr__(self):
        return "{} ({})".format(self.__class__.__name__, self.info)

    def __str__(self):
        return self.__repr__()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @property
    @abstractmethod
    def info(self):
        """Returns a unique robot identifier string.
        """
        pass
    
    @property
    @abstractmethod
    def axes(self):
        """Returns the Euler axes used to specify frames and poses.

        Allowable values:
        'sxyz', 'sxyx', 'sxzy', 'sxzx', 'syzx', 'syzy',
        'syxz', 'syxy', 'szxy', 'szxz', 'szyx', 'szyz',
        'rzyx', 'rxyx', 'ryzx', 'rxzx', 'rxzy', 'ryzy',
        'rzxy', 'ryxy', 'ryxz', 'rzxz', 'rxyz', 'rzyz'

        'r' = rotating/intrinsic/local frame
        's' = static/extrinsic/global frame
        """
        pass

    @axes.setter
    @abstractmethod
    def axes(self, axes):
        pass

    @property
    @abstractmethod
    def tcp(self):
        """Returns the tool center point (TCP) of the robot.

        The TCP is specified in the output flange frame, which is located at
        the intersection of the tool flange center axis and the flange face,
        with the z-axis aligned with the tool flange center axis.

        tool = (x, y, z, alpha, beta, gamma)
        x, y, z specify a Cartesian position
        alpha, beta, gamma specify an euler rotation
        """
        pass

    @tcp.setter
    @abstractmethod
    def tcp(self, tcp):
        """Sets the tool center point (TCP) of the robot.

        The TCP is specified in the output flange frame, which is located at
        the intersection of the tool flange center axis and the flange face,
        with the z-axis aligned with the tool flange center axis.

        tcp = (x, y, z, alpha, beta, gamma)
        x, y, z specify a Cartesian position
        alpha, beta, gamma specify an euler rotation
        """
        pass

    @property
    @abstractmethod
    def coord_frame(self):
        """Returns the reference coordinate frame for the robot.

        frame = (x, y, z, alpha, beta, gamma)
        x, y, z specify a Cartesian position
        alpha, beta, gamma specify an euler rotation
        """
        pass

    @coord_frame.setter
    @abstractmethod
    def coord_frame(self, frame):
        """Sets the reference coordinate frame for the robot.

        frame = (x, y, z, alpha, beta, gamma)
        x, y, z specify a Cartesian position
        alpha, beta, gamma specify an euler rotation
        """
        pass

    @property
    @abstractmethod
    def linear_speed(self):
        """Returns the linear speed of the robot TCP.
        """
        pass

    @linear_speed.setter
    @abstractmethod
    def linear_speed(self, speed):
        """Sets the linear speed of the robot TCP.
        """
        pass

    @property
    @abstractmethod
    def angular_speed(self):
        """Returns the angular speed of the robot TCP.
        """
        pass

    @angular_speed.setter
    @abstractmethod
    def angular_speed(self, speed):
        """Sets the angular speed of the robot TCP.
        """
        pass

    @property
    @abstractmethod
    def speed(self):
        """Returns the global speed of the robot TCP.
        """
        pass

    @speed.setter
    @abstractmethod
    def speed(self, speed):
        """Sets the global speed of the robot TCP.
        """
        pass

    @property
    @abstractmethod
    def blend_radius(self):
        """Returns the robot blend radius.
        """
        pass

    @blend_radius.setter
    @abstractmethod
    def blend_radius(self, blend_radius):
        """Sets the robot blend radius.
        """
        pass

    @property
    @abstractmethod
    def joint_angles(self):
        """ Returns the robot joint angles.

        joint_angles = (j0, j1, j2, j3, j4, j5)
        j0, j1, j2, j3, j4, j5 specify the joint angles, numbered
        from base to end effector
        """
        pass

    @property
    @abstractmethod
    def target_joint_angles(self):
        """ Returns the target robot joint angles.

        joint_angles = (j0, j1, j2, j3, j4, j5)
        j0, j1, j2, j3, j4, j5 specify the joint angles, numbered
        from base to end effector
        """
        pass

    @property
    @abstractmethod
    def pose(self):
        """Returns the TCP pose in the reference coordinate frame.

        pose = (x, y, z, alpha, beta, gamma)
        x, y, z specify a Cartesian position
        alpha, beta, gamma specify an Euler rotation
        """
        pass

    @property
    @abstractmethod
    def target_pose(self):
        """Returns the target TCP pose in the reference coordinate frame.

        pose = (x, y, z, alpha, beta, gamma)
        x, y, z specify a Cartesian position
        alpha, beta, gamma specify an Euler rotation
        """
        pass

    @property
    @abstractmethod
    def elbow(self):
        """Returns the current elbow angle.
        """
        pass

    @property
    @abstractmethod
    def target_elbow(self):
        """Returns the target elbow angle.
        """
        pass

    @abstractmethod
    def move_joints(self, joint_angles):
        """Executes an immediate move to the specified joint angles.

        joint_angles = (j0, j1, j2, j3, j4, j5)
        j0, j1, j2, j3, j4, j5 specify the joint angles, numbered
        from base to end effector
        """
        pass

    @abstractmethod
    def move_linear(self, pose, elbow):
        """Executes a linear/cartesian move from the current TCP pose to the
        specified pose in the reference coordinate frame.

        pose = (x, y, z, alpha, beta, gamma)
        x, y, z specify a Cartesian position
        alpha, beta, gamma specify an Euler rotation
        elbow = target elbow angle for 7-DOF robot arm (optional)
        """
        pass

    @abstractmethod
    def move_circular(self, via_pose, end_pose, elbow):
        """Executes a movement in a circular path from the current TCP pose,
        through via_pose, to end_pose in the reference coordinate frame.

        via_pose, end_pose = (x, y, z, alpha, beta, gamma)
        x, y, z specify a Cartesian position
        alpha, beta, gamma specify an Euler rotation
        elbow = target elbow angle for 7-DOF robot arm (optional)
        """
        pass

    @abstractmethod
    def close(self):
        """Releases any resources held by the robot (e.g., sockets).
        """
        pass


class SyncRobot(Robot):
    """Synchronous robot class provides synchronous (blocking) movement
    primitives.
    """
    
    def __init__(self, controller):
        self.controller = controller
        try:
            self.axes = 'sxyz'
            self.coord_frame = (0, 0, 0, 0, 0, 0)   # base frame (euler)
            self.tcp = (0, 0, 0, 0, 0, 0)           # tool flange frame (euler)
        except:
            self.controller.close()
            raise

    @property
    def info(self):
        """Returns a unique robot identifier string.
        """
        return self.controller.info

    @property
    def axes(self):
        """Returns the Euler axes used to specify frames and poses.
        """
        return self._axes
    
    @axes.setter
    def axes(self, axes):
        if axes not in self.EULER_AXES:
            raise InvalidEulerAxes
        self._axes = axes

    @property
    def tcp(self):
        """Returns the tool center point (TCP) of the robot. Definition in Robot class.
        """
        return quat2euler(self._tcp_q, self._axes)

    @tcp.setter
    def tcp(self, tcp):
        """Set the software tool offset, without writing controller parameters.

        With Topstar TF0 this is the tool pose relative to the flange. A
        controller-managed Topstar tool must use an identity software offset.
        """
        check_pose(tcp)
        tcp_q = np.array(euler2quat(tcp, self._axes))
        if not np.all(np.isfinite(tcp_q)):
            raise InvalidEulerPose("TCP must contain finite numbers")
        identity = np.allclose(tcp_q[:3], 0, atol=1e-12, rtol=0) and np.allclose(
            np.abs(tcp_q[3:]), (1, 0, 0, 0), atol=1e-12, rtol=0)
        if not getattr(self.controller, "supports_software_tcp", True) and not identity:
            raise ValueError("Use identity robot.tcp with a controller-managed tool; "
                             "use Topstar tool_index=0 for a software TCP")
        self._tcp_q = tcp_q
        
    @property
    def tcp_controller(self):
        """Returns the tool center point (TCP) of the robot.
        Version that points to Robot Controller/API.
        """
        return quat2euler(self.controller.tcp, self._axes)

    @tcp_controller.setter
    def tcp_controller(self, tcp):
        """Sets the tool center point (TCP) of the robot.
        Version that points to Robot Controller/API.
        """
        check_pose(tcp)
        self.controller.tcp = euler2quat(tcp, self._axes)

    @property
    def coord_frame(self):
        """Returns the reference coordinate frame for the robot TCP.
        """
        return quat2euler(self._coord_frame_q, self._axes)

    @coord_frame.setter
    def coord_frame(self, frame):
        """Sets the reference coordinate frame for the robot TCP.
        """
        check_pose(frame)
        self._coord_frame_q = euler2quat(frame, self._axes)

    @property
    def linear_speed(self):
        """Returns the linear speed of the robot TCP.
        """
        return self.controller.linear_speed

    @linear_speed.setter
    def linear_speed(self, speed):
        """Sets the linear speed of the robot TCP.
        """
        self.controller.linear_speed = speed

    @property
    def angular_speed(self):
        """Returns the angular speed of the robot TCP.
        """
        return self.controller.angular_speed

    @angular_speed.setter
    def angular_speed(self, speed):
        """Sets the angular speed of the robot TCP.
        """
        self.controller.angular_speed = speed

    @property
    def speed(self):
        """Returns the speed of the robot TCP.
        """
        return self.controller.speed
    
    @speed.setter
    def speed(self, speed):
        """Sets the global speed of the robot TCP.
        """
        self.controller.speed = speed
    
    @property
    def blend_radius(self):
        """Returns the robot blend radius (mm).
        """
        return self.controller.blend_radius

    @blend_radius.setter
    def blend_radius(self, blend_radius):
        """Sets the robot blend radius (mm).
        """
        self.controller.blend_radius = blend_radius

    @property
    def command_speed(self):
        """Backend per-command speed percentage."""
        return self.controller.command_speed

    @command_speed.setter
    def command_speed(self, speed):
        if not hasattr(self.controller, "command_speed"):
            raise NotImplementedError("Controller has no command_speed percentage interface")
        self.controller.command_speed = speed

    @property
    def joint_angles(self):
        """ Returns the current joint angles.
        """
        return self.controller.joint_angles

    @property
    def target_joint_angles(self):
        """ Returns the target joint angles.
        """
        return self.controller.commanded_joint_angles

    @property
    def pose(self):
        """Returns the current TCP pose in the reference coordinate frame.
        """
        base_pose_q = self.controller.pose
        pose_q = transform(base_pose_q, self._coord_frame_q)
        tcp_pose_q = inv_transform(self._tcp_q, pose_q)
        return quat2euler(tcp_pose_q, self._axes)
    
    @property
    def target_pose(self):
        """Returns the target TCP pose in the reference coordinate frame.
        """
        base_pose_q = self.controller.commanded_pose
        pose_q = transform(base_pose_q, self._coord_frame_q)
        tcp_pose_q = inv_transform(self._tcp_q, pose_q)
        return quat2euler(tcp_pose_q, self._axes)

    @property
    def elbow(self):
        """Returns the current elbow angle (degrees).
        """
        return self.controller.elbow

    @property
    def target_elbow(self):
        """Returns the target elbow angle (degrees).
        """
        return self.controller.commanded_elbow

    def move_joints(self, joint_angles):
        """Executes an immediate move to the specified joint angles.
        """
        check_joint_angles(joint_angles)
        joint_angles = np.array(joint_angles)
        self.controller.move_joints(joint_angles)
    
    def move_linear(self, pose, elbow=None, *, wrist=0):
        """Executes a linear/cartesian move from the current TCP pose to the
        specified pose in the reference coordinate frame.
        """
        target = self._to_controller_pose(pose)
        if wrist == 0:
            self.controller.move_linear(target, elbow)
        elif wrist == 1:
            self.controller.move_linear(target, elbow, wrist=wrist)
        else:
            raise ValueError("wrist must be 0 or 1")

    def check_linear_targets(self, poses):
        """Check consecutive TCP targets using the controller's read-only IK."""
        self._controller_call("check_linear_targets", [self._to_controller_pose(pose) for pose in poses])

    @contextmanager
    def linear_path_checker(self):
        """Check successive TCP target groups without moving the robot."""
        with self._controller_call("linear_path_checker") as check:
            def check_tcp(poses):
                check([self._to_controller_pose(pose) for pose in poses])
            yield check_tcp

    def _to_controller_pose(self, pose):
        check_pose(pose)
        pose_q = euler2quat(pose, self._axes)
        endpoint_pose_q = frame(pose_q, self._tcp_q)
        return inv_transform(endpoint_pose_q, self._coord_frame_q)

    def move_circular(self, via_pose, end_pose, elbow=None):
        """Executes a movement in a circular path from the current TCP pose,
        through via_pose, to end_pose in the reference coordinate frame.
        """
        self.controller.move_circular(self._to_controller_pose(via_pose),
                                      self._to_controller_pose(end_pose), elbow)

    def _controller_call(self, name, *args, **kwargs):
        method = getattr(self.controller, name, None)
        if method is None:
            raise NotImplementedError("{} does not support {}".format(
                type(self.controller).__name__, name))
        return method(*args, **kwargs)

    def prepare_motion(self):
        return self._controller_call("prepare_motion")

    def prepare_servo(self, **options):
        """Explicitly begin native servo submission; see the backend contract."""
        return self._controller_call("prepare_servo", **options)

    def servo_move_linear(self, pose, elbow=None):
        """Submit a servo target; return does not mean the target was reached."""
        return self._controller_call("servo_move_linear", self._to_controller_pose(pose), elbow)

    def servo_move_joints(self, joint_angles):
        check_joint_angles(joint_angles)
        return self._controller_call("servo_move_joints", np.array(joint_angles))

    def stop(self):
        return self._controller_call("stop")

    def abort(self):
        return self._controller_call("abort")

    def disable_servo(self):
        return self._controller_call("disable_servo")

    def close(self):
        """Releases any resources held by the robot (e.g., sockets).
        """
        self.controller.close()      


class AsyncRobot(Robot):
    """Asynchronous robot class provides synchronous (blocking) and
    asynchronous (non-blocking) movement primitives.
    """
    
    def __init__(self, sync_robot, *, close_timeout=5):
        self.sync_robot = sync_robot
        try:
            if not np.isfinite(close_timeout) or close_timeout <= 0:
                raise ValueError("close_timeout must be positive and finite")
            self._close_timeout = float(close_timeout)
            self._async_lock = Lock()
            self._worker = None
            self._future = None
            self._busy = False
            self._closed = False
        except:
            self.sync_robot.close()
            raise

    @property
    def info(self):
        return self.sync_robot.info

    @property
    def axes(self):
        return self.sync_robot.axes

    @axes.setter
    def axes(self, axes):
        self.sync_robot.axes = axes

    @property
    def tcp(self):
        return self.sync_robot.tcp

    @tcp.setter
    def tcp(self, tcp):
        self.sync_robot.tcp = tcp

    @property
    def tcp_controller(self):
        return self.sync_robot.tcp_controller

    @tcp_controller.setter
    def tcp_controller(self, tcp):
        self.sync_robot.tcp_controller = tcp

    @property
    def coord_frame(self):
        return self.sync_robot.coord_frame

    @coord_frame.setter
    def coord_frame(self, frame):
        self.sync_robot.coord_frame = frame

    @property
    def linear_speed(self):
        return self.sync_robot.linear_speed

    @linear_speed.setter
    def linear_speed(self, speed):
        self.sync_robot.linear_speed = speed

    @property
    def angular_speed(self):
        return self.sync_robot.angular_speed

    @angular_speed.setter
    def angular_speed(self, speed):
        self.sync_robot.angular_speed = speed

    @property
    def speed(self):
        return self.sync_robot.speed

    @speed.setter
    def speed(self, speed):
        self.sync_robot.speed = speed

    @property
    def blend_radius(self):
        return self.sync_robot.blend_radius

    @blend_radius.setter
    def blend_radius(self, blend_radius):
        self.sync_robot.blend_radius = blend_radius

    @property
    def command_speed(self):
        return self.sync_robot.command_speed

    @command_speed.setter
    def command_speed(self, speed):
        self.sync_robot.command_speed = speed

    @property
    def joint_angles(self):
        return self.sync_robot.joint_angles

    @property
    def target_joint_angles(self):
        return self.sync_robot.target_joint_angles

    @property
    def pose(self):
        return self.sync_robot.pose

    @property
    def target_pose(self):
        return self.sync_robot.target_pose

    @property
    def elbow(self):
        return self.sync_robot.elbow

    @property
    def target_elbow(self):
        return self.sync_robot.target_elbow

    def move_joints(self, joint_angles):
        self.sync_robot.move_joints(joint_angles)

    def move_linear(self, pose, elbow=None):
        self.sync_robot.move_linear(pose, elbow)

    def move_circular(self, via_pose, end_pose, elbow=None):
        self.sync_robot.move_circular(via_pose, end_pose, elbow)

    def prepare_motion(self):
        return self.sync_robot.prepare_motion()

    def prepare_servo(self, **options):
        return self.sync_robot.prepare_servo(**options)

    def servo_move_linear(self, pose, elbow=None):
        return self.sync_robot.servo_move_linear(pose, elbow)

    def servo_move_joints(self, joint_angles):
        return self.sync_robot.servo_move_joints(joint_angles)

    def stop(self):
        return self.sync_robot.stop()

    def abort(self):
        return self.sync_robot.abort()

    def disable_servo(self):
        return self.sync_robot.disable_servo()

    def _start_async(self, method, *args):
        with self._async_lock:
            if self._closed:
                raise RuntimeError("Robot is closed")
            if self._busy:
                raise AsyncBusy
            future = Future()

            def run():
                try:
                    future.set_result(method(*args))
                except BaseException as error:
                    future.set_exception(error)

            worker = Thread(target=run, daemon=True)
            self._future, self._worker, self._busy = future, worker, True
            try:
                worker.start()
            except BaseException:
                self._future, self._worker, self._busy = None, None, False
                raise

    def async_move_joints(self, joint_angles):
        """Executes an immediate move to the specified joint angles.
        
        joint_angles = (j0, j1, j2, j3, j4, j5)
        j0, j1, j2, j3, j4, j5 specify the joint angles (degrees), numbered
        from base to end effector
        """
        self._start_async(self.sync_robot.move_joints, np.array(joint_angles, copy=True))
    
    def async_move_linear(self, pose, elbow=None):
        """Executes a linear/cartesian move from the current TCP pose to the
        specified pose in the reference coordinate frame.
        
        pose = (x, y, z, alpha, beta, gamma)
        x, y, z specify a Cartesian position (mm)
        alpha, beta, gamma specify an euler rotation (degrees)
        elbow = target elbow angle for 7-DOF robot arms (optional; degrees)
        """
        self._start_async(self.sync_robot.move_linear, np.array(pose, copy=True), elbow)

    def async_move_circular(self, via_pose, end_pose, elbow=None):
        """Executes a movement in a circular path from the current TCP pose,
        through via_pose, to end_pose in the reference coordinate frame.

        via_pose, end_pose = (x, y, z, alpha, beta, gamma)
        x, y, z specify a Cartesian position (mm)
        alpha, beta, gamma specify an euler rotation (degrees)
        elbow = target elbow angle for 7-DOF robot arms (optional; degrees)
        """
        self._start_async(self.sync_robot.move_circular,
                          np.array(via_pose, copy=True), np.array(end_pose, copy=True), elbow)

    def async_servo_move_linear(self, pose, elbow=None):
        """Run one native servo submission in a worker; consume async_result()."""
        self._start_async(self.sync_robot.servo_move_linear, np.array(pose, copy=True), elbow)

    def async_servo_move_joints(self, joint_angles):
        self._start_async(self.sync_robot.servo_move_joints, np.array(joint_angles, copy=True))
    
    def async_result(self, timeout=None):
        """Wait for a submission/move and re-raise its original exception.

        A timeout leaves the result pending so it can be collected later.
        """
        with self._async_lock:
            if not self._busy:
                raise AsyncNotBusy
            worker, future = self._worker, self._future
        worker.join(timeout)
        if worker.is_alive():
            raise TimeoutError("Asynchronous operation is still running")
        try:
            return future.result()
        finally:
            with self._async_lock:
                if self._future is future:
                    self._busy = False
                    self._worker = self._future = None
    
    def async_done(self):
        """Checks to see whether asynchronous operation has finished.
        """
        with self._async_lock:
            if not self._busy:
                raise AsyncNotBusy
            return not self._worker.is_alive()
        
    def close(self):
        """Releases any resources held by the robot (e.g., sockets).
        """
        with self._async_lock:
            if self._closed:
                return
            self._closed = True
            worker = self._worker
        try:
            self.sync_robot.close()
        finally:
            if worker is not None:
                worker.join(self._close_timeout)
                if worker.is_alive():
                    raise TimeoutError("Robot closed but the SDK worker has not returned")
