"""RGB-D-only TraX entry using the same semantic runtime as OPE."""
import argparse

from semantic_runtime import checked_plan, make_tracker, text_bank


def run(plan_path):
    plan, bundle = checked_plan(plan_path)
    bank = text_bank(plan, bundle)
    tracker = make_tracker(bundle)
    import m39_vot_bridge as vot
    from lib.train.dataset.depth_utils import get_rgbd_frame
    handle = vot.VOT('rectangle', channels='rgbd')
    bbox = list(handle.region())
    files = handle.frame()
    assert isinstance(files, list) and len(files) == 2
    image = get_rgbd_frame(files[0], files[1], dtype='rgbcolormap', depth_clip=True)
    tracker.initialize(image, bank.info(files[0], bbox))
    while True:
        files = handle.frame()
        if files is None:
            break
        assert isinstance(files, list) and len(files) == 2
        image = get_rgbd_frame(files[0], files[1], dtype='rgbcolormap', depth_clip=True)
        prediction = tracker.track(image)
        handle.report(vot.Rectangle(*prediction['target_bbox']), prediction['best_score'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    run(parser.parse_args().plan)
