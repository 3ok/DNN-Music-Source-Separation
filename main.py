from models import *

# preparing the data
dsd = dsdtools.DB(PATH_DATA)
tracks = dsd.load_dsd_tracks()
train_tracks, test_tracks = tracks[:50], tracks[50:]

# processing tracks and targets
train_features = process_all_tracks(train_tracks)
train_targets = dict()
my_models = dict()
for target_name in TARGET_NAMES:
    train_targets[target_name] = process_target(train_tracks, target_name)
    my_models[target_name] = full_model(target_name)




