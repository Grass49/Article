# npu_connector.py
# 管理本地Docker到昇腾310B的SSH连接
# 依赖：pip install paramiko

import os
import threading
import time

import paramiko


class NPUConnector:
    def __init__(self, host, port=22, username="root", password=None, key_path=None):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.key_path = key_path
        self._client = None
        # RLock允许exec/upload/download在持锁状态下触发自动重连。
        self._lock = threading.RLock()

    def _is_connected(self):
        """Return True only when the SSH transport is active and authenticated."""
        if self._client is None:
            return False
        transport = self._client.get_transport()
        return bool(
            transport is not None
            and transport.is_active()
            and transport.is_authenticated()
        )

    def connect(self, announce=True):
        """建立SSH连接，并开启keepalive以降低ATC编译期间空闲断线概率。"""
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass

            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            common = {
                "hostname": self.host,
                "port": self.port,
                "username": self.username,
                "timeout": 15,
                "banner_timeout": 15,
                "auth_timeout": 15,
            }
            if self.key_path:
                common["key_filename"] = self.key_path
            else:
                common["password"] = self.password

            client.connect(**common)
            transport = client.get_transport()
            if transport is not None:
                # ATC编译可能持续较久，期间主SSH连接没有业务流量。
                transport.set_keepalive(20)

            self._client = client

        if announce:
            print(f"[NPU] 已连接到 {self.host}:{self.port}")

    def ensure_connected(self):
        """确保SSH可用；失效时自动重新连接。"""
        with self._lock:
            if self._is_connected():
                return
            print("[NPU] SSH连接失效，正在重新连接...")
            self.connect(announce=False)
            print(f"[NPU] 已重新连接到 {self.host}:{self.port}")

    def _reconnect_after_failure(self, exc):
        """远程操作失败后强制重连。"""
        print(f"[NPU] SSH操作失败({type(exc).__name__}: {exc!r})，尝试重新连接...")
        # 给嵌入式板卡的sshd一点清理旧会话的时间。
        time.sleep(0.3)
        self.connect(announce=False)

    def clone(self):
        """Create an independent SSH connection for concurrent monitoring."""
        cloned = NPUConnector(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            key_path=self.key_path,
        )
        cloned.connect(announce=False)
        return cloned

    def exec(self, cmd, timeout=120):
        """执行远程命令，返回(stdout, stderr, returncode)；断线时自动重试。"""
        with self._lock:
            self.ensure_connected()
            last_exc = None
            for attempt in range(3):
                try:
                    _, stdout, stderr = self._client.exec_command(cmd, timeout=timeout)
                    out = stdout.read().decode(errors="replace")
                    err = stderr.read().decode(errors="replace")
                    rc = stdout.channel.recv_exit_status()
                    return out, err, rc
                except (paramiko.SSHException, EOFError, OSError) as exc:
                    last_exc = exc
                    if attempt == 2:
                        break
                    self._reconnect_after_failure(exc)
            raise last_exc

    @staticmethod
    def _upload_temp_path(local_path, remote_path):
        """为同一个本地文件生成稳定的断点续传临时名。"""
        stat = os.stat(local_path)
        # 同一文件在后续调用中仍会得到同一临时名；重新编译后mtime改变，避免续传旧内容。
        return "{0}.part.{1}.{2}".format(remote_path, stat.st_size, stat.st_mtime_ns)

    def upload(self, local_path, remote_path, chunk_size=256 * 1024, max_attempts=8):
        """可靠上传文件到310B。

        使用分块断点续传到临时文件；只有远端大小与本地完全一致后，
        才原子地替换正式文件。这样即使SFTP中途掉线，也不会留下一个
        被后续msame误认为有效模型的残缺.om文件。
        """
        if not os.path.isfile(local_path):
            raise FileNotFoundError(local_path)

        local_size = os.path.getsize(local_path)
        temp_path = self._upload_temp_path(local_path, remote_path)

        with self._lock:
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                self.ensure_connected()
                sftp = None
                remote_file = None
                try:
                    sftp = self._client.open_sftp()

                    try:
                        remote_size = int(sftp.stat(temp_path).st_size)
                    except OSError:
                        remote_size = 0

                    if remote_size < 0 or remote_size > local_size:
                        try:
                            sftp.remove(temp_path)
                        except OSError:
                            pass
                        remote_size = 0

                    if remote_size:
                        print(
                            "[NPU] 续传: {0}，已完成 {1}/{2} bytes".format(
                                remote_path, remote_size, local_size
                            )
                        )

                    mode = "r+b" if remote_size else "wb"
                    remote_file = sftp.file(temp_path, mode)
                    # 关闭写流水线，降低310B端SSH/SFTP在大文件连续写入时的压力。
                    if hasattr(remote_file, "set_pipelined"):
                        remote_file.set_pipelined(False)
                    if remote_size:
                        remote_file.seek(remote_size)

                    with open(local_path, "rb") as local_file:
                        local_file.seek(remote_size)
                        while True:
                            block = local_file.read(chunk_size)
                            if not block:
                                break
                            remote_file.write(block)
                            # 每块确认，断线后stat得到的是已真正提交的偏移。
                            remote_file.flush()

                    remote_file.close()
                    remote_file = None

                    uploaded_size = int(sftp.stat(temp_path).st_size)
                    if uploaded_size != local_size:
                        raise IOError(
                            "上传大小校验失败: remote={0}, local={1}".format(
                                uploaded_size, local_size
                            )
                        )

                    # 只有完整临时文件才替换正式文件；正式路径永远不暴露半成品。
                    try:
                        sftp.remove(remote_path)
                    except OSError:
                        pass
                    sftp.rename(temp_path, remote_path)

                    final_size = int(sftp.stat(remote_path).st_size)
                    if final_size != local_size:
                        raise IOError(
                            "远端文件大小校验失败: remote={0}, local={1}".format(
                                final_size, local_size
                            )
                        )

                    print(
                        "[NPU] 上传完成: {0} → {1} ({2} bytes)".format(
                            local_path, remote_path, local_size
                        )
                    )
                    return

                except (paramiko.SSHException, EOFError, OSError, IOError) as exc:
                    last_exc = exc
                    if attempt >= max_attempts:
                        break
                    print(
                        "[NPU] 上传中断，第 {0}/{1} 次尝试失败: {2}: {3!r}".format(
                            attempt, max_attempts, type(exc).__name__, exc
                        )
                    )
                    self._reconnect_after_failure(exc)
                finally:
                    if remote_file is not None:
                        try:
                            remote_file.close()
                        except Exception:
                            pass
                    if sftp is not None:
                        try:
                            sftp.close()
                        except Exception:
                            pass

            raise RuntimeError(
                "文件上传在{0}次尝试后仍失败: {1} -> {2}; last_error={3}: {4!r}".format(
                    max_attempts,
                    local_path,
                    remote_path,
                    type(last_exc).__name__ if last_exc else "Unknown",
                    last_exc,
                )
            )

    def download(self, remote_path, local_path):
        """从310B下载文件；断线时自动重试。"""
        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)

        with self._lock:
            self.ensure_connected()
            last_exc = None
            for attempt in range(3):
                sftp = None
                try:
                    sftp = self._client.open_sftp()
                    sftp.get(remote_path, local_path)
                    print(f"[NPU] 下载: {remote_path} → {local_path}")
                    return
                except (paramiko.SSHException, EOFError, OSError) as exc:
                    last_exc = exc
                    if attempt == 2:
                        break
                    self._reconnect_after_failure(exc)
                finally:
                    if sftp is not None:
                        try:
                            sftp.close()
                        except Exception:
                            pass
            raise last_exc

    def close(self):
        with self._lock:
            if self._client:
                self._client.close()
                self._client = None
