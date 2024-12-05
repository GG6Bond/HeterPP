# from .dist_gpipe_pipeline_async_bk import GpipeAsync
from .dist_gpipe_pipeline_async import GpipeAsync
from .dist_1f1b_pipeline_async import Async1F1B
from .dtfm_gpipe_async import DTFMAsyncGPipe
from .dtfm_1f1b_async import DTFMAsync1F1B


def get_pp_module(args, vocab_size, num_classes, device, use_dp):
    if args.pp_mode == 'gpipe':
        return GpipeAsync(args, vocab_size, num_classes, device, use_dp)
    if args.pp_mode == '1f1b':
        return Async1F1B(args, vocab_size, num_classes, device, use_dp)
    if args.pp_mode == 'dtfm_gpipe':
        return DTFMAsyncGPipe(args, vocab_size, num_classes, device, use_dp)
    if args.pp_mode == 'dtfm_1f1b':
        return DTFMAsync1F1B(args, vocab_size, num_classes, device, use_dp)
    else:
        print("Not recognize this pipeline parallel mode.")
        assert False
