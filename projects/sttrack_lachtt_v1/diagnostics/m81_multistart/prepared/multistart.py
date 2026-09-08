"""Initialize only a predeclared training/development episode."""
def initialize_episode(tracker, bank, episode, image):
    index = bank['ids'].index(episode['id'])
    tracker.initialize(image, dict(init_bbox=episode['init_bbox'], text_tokens=bank['tokens'][index],
        text_mask=bank['mask'][index], empty_text=bank['empty']))
    assert tracker.frame_id == 0
    return index
