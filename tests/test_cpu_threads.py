from expds import cpu_threads


def test_physical_cores_sane(monkeypatch):
    import os
    monkeypatch.delenv("EXPDS_CPU_THREADS", raising=False)
    n = cpu_threads.physical_cores()
    assert 1 <= n <= (os.cpu_count() or 1)
    monkeypatch.setenv("EXPDS_CPU_THREADS", "3")
    assert cpu_threads.physical_cores() == 3


def test_torch_all_cores_restores():
    import torch
    old = torch.get_num_threads()
    with cpu_threads.torch_all_cores():
        assert torch.get_num_threads() == cpu_threads.physical_cores()
    assert torch.get_num_threads() == old
