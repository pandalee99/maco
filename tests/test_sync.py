"""
MACO Synchronization Module Tests

测试 Stream Manager 和 Signal-Wait 机制。
"""

import pytest
import torch
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cuda_available():
    """检查 CUDA 是否可用"""
    return torch.cuda.is_available()


# ========== StreamManager Tests ==========


@pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
class TestStreamManager:
    """StreamManager 测试"""
    
    def setup_method(self):
        """每个测试前重置"""
        from maco.sync import StreamManager
        StreamManager.reset()
    
    def test_singleton_per_device(self):
        """每个设备一个实例"""
        from maco.sync import StreamManager
        
        sm1 = StreamManager()
        sm2 = StreamManager()
        
        assert sm1 is sm2
    
    def test_compute_stream_created(self):
        """计算 stream 正确创建"""
        from maco.sync import StreamManager
        
        sm = StreamManager()
        assert sm.compute_stream is not None
        assert isinstance(sm.compute_stream, torch.cuda.Stream)
    
    def test_comm_stream_created(self):
        """通信 stream 正确创建"""
        from maco.sync import StreamManager
        
        sm = StreamManager()
        assert sm.comm_stream is not None
        assert isinstance(sm.comm_stream, torch.cuda.Stream)
    
    def test_streams_are_different(self):
        """计算和通信 stream 不同"""
        from maco.sync import StreamManager
        
        sm = StreamManager()
        # 不同的 stream 对象
        assert sm.compute_stream is not sm.comm_stream
    
    def test_compute_wait_comm(self):
        """计算 stream 等待通信 stream"""
        from maco.sync import StreamManager
        
        sm = StreamManager()
        
        # 在通信 stream 上执行操作
        result = torch.zeros(1000, 1000, device="cuda")
        with torch.cuda.stream(sm.comm_stream):
            for _ in range(10):
                result = torch.matmul(result, result.T)
        
        # 让计算 stream 等待
        sm.compute_wait_comm()
        
        # 在计算 stream 上应该能看到结果
        with torch.cuda.stream(sm.compute_stream):
            assert result.shape == (1000, 1000)
    
    def test_comm_wait_compute(self):
        """通信 stream 等待计算 stream"""
        from maco.sync import StreamManager
        
        sm = StreamManager()
        
        # 在计算 stream 上执行操作
        result = torch.zeros(1000, 1000, device="cuda")
        with torch.cuda.stream(sm.compute_stream):
            for _ in range(10):
                result = torch.matmul(result, result.T)
        
        # 让通信 stream 等待
        sm.comm_wait_compute()
        
        # 在通信 stream 上应该能看到结果
        with torch.cuda.stream(sm.comm_stream):
            assert result.shape == (1000, 1000)


# ========== Signal Tests ==========


@pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
class TestSignal:
    """Signal 测试"""
    
    def test_create_signal(self):
        """创建信号"""
        from maco.sync import create_signal
        
        sig = create_signal("test_signal")
        assert sig.name == "test_signal"
        assert not sig.signaled
    
    def test_signal_record(self):
        """记录信号"""
        from maco.sync import create_signal
        
        sig = create_signal("test")
        stream = torch.cuda.Stream()
        
        with torch.cuda.stream(stream):
            # 执行一些操作
            _ = torch.randn(100, 100, device="cuda")
        
        sig.record(stream)
        assert sig.signaled
        assert sig.stream is stream
    
    def test_signal_wait(self):
        """等待信号"""
        from maco.sync import create_signal
        
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        
        sig = create_signal("test")
        
        # 在 stream1 上执行并记录信号
        with torch.cuda.stream(stream1):
            _ = torch.randn(1000, 1000, device="cuda")
        sig.record(stream1)
        
        # 在 stream2 上等待
        sig.wait(stream2)
        
        # 同步 stream2
        stream2.synchronize()
    
    def test_signal_is_ready(self):
        """检查信号就绪"""
        from maco.sync import create_signal
        
        sig = create_signal("test")
        
        # 未发出信号
        assert not sig.is_ready()
        
        # 发出信号
        sig.record()
        torch.cuda.synchronize()
        
        # 应该就绪
        assert sig.is_ready()


# ========== SignalWait Tests ==========


@pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
class TestSignalWait:
    """SignalWait 测试"""
    
    def test_create_named_signal(self):
        """创建命名信号"""
        from maco.sync import SignalWait
        
        sw = SignalWait()
        sig = sw.create_signal("compute_done")
        
        assert sig.name == "compute_done"
        assert sw.get_signal("compute_done") is sig
    
    def test_signal_and_wait(self):
        """发信号和等待"""
        from maco.sync import SignalWait
        
        sw = SignalWait()
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        
        # 在 stream1 上发信号
        with torch.cuda.stream(stream1):
            _ = torch.randn(100, 100, device="cuda")
        sw.signal("done", stream1)
        
        # 在 stream2 上等待
        sw.wait("done", stream2)
        stream2.synchronize()
    
    def test_wave_signals(self):
        """Wave 信号"""
        from maco.sync import SignalWait
        
        sw = SignalWait()
        stream = torch.cuda.Stream()
        
        # 发出多个 wave 信号
        for i in range(3):
            with torch.cuda.stream(stream):
                _ = torch.randn(100, 100, device="cuda")
            sw.signal_wave(i, stream)
        
        assert sw.num_waves == 3
        
        # 等待所有 wave
        sw.wait_all_waves()
    
    def test_reset(self):
        """重置信号"""
        from maco.sync import SignalWait
        
        sw = SignalWait()
        sw.create_signal("test")
        sw.signal_wave(0)
        
        sw.reset()
        
        assert sw.get_signal("test") is None
        assert sw.num_waves == 0


# ========== OverlapContext Tests ==========


@pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
class TestOverlapContext:
    """OverlapContext 测试"""
    
    def test_create_context(self):
        """创建上下文"""
        from maco.sync import OverlapContext
        
        ctx = OverlapContext()
        assert ctx.compute_stream is not None
        assert ctx.comm_stream is not None
    
    def test_compute_comm_overlap(self):
        """计算-通信重叠"""
        from maco.sync import OverlapContext
        
        ctx = OverlapContext()
        
        # Wave 0: 计算
        with ctx.compute():
            result_0 = torch.randn(1000, 1000, device="cuda")
            result_0 = torch.matmul(result_0, result_0.T)
        ctx.signal_compute_done(0)
        
        # Wave 0: 通信（模拟）
        with ctx.comm():
            ctx.wait_compute_done(0)
            # 模拟通信：复制数据
            comm_result = result_0.clone()
        ctx.signal_comm_done(0)
        
        # Wave 1: 计算（与 Wave 0 通信重叠）
        with ctx.compute():
            result_1 = torch.randn(1000, 1000, device="cuda")
            result_1 = torch.matmul(result_1, result_1.T)
        ctx.signal_compute_done(1)
        
        # 等待通信完成
        ctx.wait_comm_done(0)
        
        ctx.sync_all()
        
        assert result_0.shape == (1000, 1000)
        assert result_1.shape == (1000, 1000)
    
    def test_multiple_waves(self):
        """多 wave 重叠"""
        from maco.sync import OverlapContext
        
        ctx = OverlapContext()
        results = []
        
        num_waves = 4
        
        for wave_id in range(num_waves):
            # 计算
            with ctx.compute():
                if wave_id > 0:
                    ctx.wait_comm_done(wave_id - 1)
                result = torch.randn(500, 500, device="cuda")
                result = torch.matmul(result, result.T)
                results.append(result)
            ctx.signal_compute_done(wave_id)
            
            # 通信（与下一波计算重叠）
            with ctx.comm():
                ctx.wait_compute_done(wave_id)
                # 模拟通信
                _ = results[-1].clone()
            ctx.signal_comm_done(wave_id)
        
        ctx.sync_all()
        
        assert len(results) == num_waves


# ========== Convenience Functions Tests ==========


@pytest.mark.skipif(not cuda_available(), reason="CUDA not available")
class TestConvenienceFunctions:
    """便捷函数测试"""
    
    def test_get_compute_stream(self):
        """获取计算 stream"""
        from maco.sync import get_compute_stream, StreamManager
        StreamManager.reset()
        
        stream = get_compute_stream()
        assert stream is not None
    
    def test_get_comm_stream(self):
        """获取通信 stream"""
        from maco.sync import get_comm_stream, StreamManager
        StreamManager.reset()
        
        stream = get_comm_stream()
        assert stream is not None


# ========== Main Entry ==========


if __name__ == "__main__":
    """直接运行测试"""
    if not cuda_available():
        print("CUDA not available, skipping tests")
        exit(0)
    
    print("Running sync module tests...")
    
    # StreamManager 测试
    from maco.sync import StreamManager
    StreamManager.reset()
    sm = StreamManager()
    assert sm.compute_stream is not None
    assert sm.comm_stream is not None
    print("  [PASS] StreamManager")
    
    # Signal 测试
    from maco.sync import create_signal
    sig = create_signal("test")
    sig.record()
    torch.cuda.synchronize()
    assert sig.is_ready()
    print("  [PASS] Signal")
    
    # SignalWait 测试
    from maco.sync import SignalWait
    sw = SignalWait()
    sw.signal("test")
    sw.wait("test")
    print("  [PASS] SignalWait")
    
    # OverlapContext 测试
    from maco.sync import OverlapContext
    ctx = OverlapContext()
    with ctx.compute():
        _ = torch.randn(100, 100, device="cuda")
    ctx.signal_compute_done(0)
    with ctx.comm():
        ctx.wait_compute_done(0)
    ctx.sync_all()
    print("  [PASS] OverlapContext")
    
    print("\n[SUCCESS] All sync tests passed!")
