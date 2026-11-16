"""Servo-off cleanup for the native X5 controller."""

import time


def _wait_for(sdk, handle, predicate, description, timeout):
    deadline = time.monotonic() + timeout
    while True:
        state = sdk.get_all_system_state(handle)
        if state.link != 1:
            raise RuntimeError("通信断开，无法确认" + description)
        if predicate(state):
            return state
        if time.monotonic() >= deadline:
            raise TimeoutError("等待{}超时".format(description))
        time.sleep(0.05)


def disable_servo(sdk, handle, *, force_stop=False, timeout=5):
    """Confirm stopped and servo-off, even after an alarm or failed motion.

    A stop failure does not suppress the attempt to disable. A successful return
    requires feedback confirming enable=0, axes stopped, and queue empty.
    """
    stop_error = None
    try:
        state = sdk.get_all_system_state(handle)
        force_stop = force_stop or state.in_pos != 1 or state.cmd_queue != 0
    except Exception as error:
        stop_error = error
        force_stop = True
    if force_stop:
        try:
            sdk.abort(handle)
            sdk.wait_cmd_send_done(handle)
            _wait_for(sdk, handle, lambda s: s.in_pos == 1 and s.cmd_queue == 0,
                     "停止", timeout=timeout)
        except Exception as error:
            stop_error = error
    try:
        try:
            state = sdk.get_all_system_state(handle)
        except Exception:
            # A feedback read can fail independently from a command send.
            # Still attempt servo-off; the SDK enforces its local-mode rule.
            state = None
        if state is not None and state.remote != 0:
            raise RuntimeError("控制柜不是本地控制，无法通过此接口下使能")
        sdk.enable_servo(handle, False)
        sdk.wait_cmd_send_done(handle)
        _wait_for(sdk, handle,
                 lambda s: s.enable == 0 and s.in_pos == 1 and s.cmd_queue == 0,
                 "伺服下使能", timeout=timeout)
    except Exception as error:
        failure = RuntimeError("未确认伺服下使能：{}".format(error))
        failure.stop_error = stop_error
        raise failure from error
