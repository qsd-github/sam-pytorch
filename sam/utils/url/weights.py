import requests
from tqdm import tqdm


class Weight:
    @staticmethod
    def download(vit_name, url, weight_path):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        # 使用 tqdm 显示下载进度
        with open(weight_path, 'wb') as file, tqdm(
                desc=vit_name + " Downloading: ",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)