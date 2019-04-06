"""
The script maps train data from

```
{data_path}/CVC-VideoClinicDBtrain_valid
    1
        1_1.png
        1_1_mask_png
        ...
```

to

```
{data_path}/train
    1
        images
            1_1.png
         masks
            1_1.png
     ...
```

"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import shutil


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-c', '--config_path', default='configs/dataset.json', type=str)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)

    data_path = Path(config['data_path']).absolute().expanduser()
    new_train_path = data_path / 'train'
    new_train_path.mkdir(exist_ok=True)

    old_train_path = data_path / 'CVC-VideoClinicDBtrain_valid'

    for video_id_path in old_train_path.glob('*'):
        video_id = video_id_path.name

        new_video_id_path = new_train_path / video_id
        new_video_id_path.mkdir(exist_ok=True)

        image_path = new_video_id_path / 'images'
        mask_path = new_video_id_path / 'masks'

        image_path.mkdir(exist_ok=True)
        mask_path.mkdir(exist_ok=True)

        for file_path in tqdm(sorted(video_id_path.glob('*.png'))):
            file_name = file_path.name

            if 'mask' in file_name:
                new_file_path = str(mask_path.joinpath(file_name.replace('_mask', '')))
            else:
                new_file_path = str(image_path.joinpath(file_name))

            shutil.copy(str(file_path), new_file_path)


if __name__ == '__main__':
    main()
