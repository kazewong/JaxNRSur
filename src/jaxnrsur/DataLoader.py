import h5py
import requests
import os


h5_mode_tuple: dict[tuple[int, int], str] = {
    (2, 0): "ITEM_6",  # No Imaginary part
    (2, 1): "ITEM_5",
    (2, 2): "ITEM_8",
    (3, 0): "ITEM_3",  # No Real part
    (3, 1): "ITEM_4",
    (3, 2): "ITEM_0",
    (3, 3): "ITEM_2",
    (4, 2): "ITEM_9",
    (4, 3): "ITEM_7",
    (4, 4): "ITEM_1",
    (5, 5): "ITEM_10",
}


def download_from_zenodo(url: str, local_filename: str) -> bool:
    """
    Helper function to download data from zenodo
    """
    # Send the HTTP request to the URL
    response = requests.get(url, stream=True)  # type: ignore

    # Check if the request was successful
    if response.status_code == 200:
        # Open a local file for writing the binary content
        with open(local_filename, "wb") as f:
            # Write the content in chunks to avoid memory overload
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
        print(f"File downloaded successfully and saved as {local_filename}")
        return True
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
        return False


def load_data(url: str, local_filename: str) -> h5py.File:
    home_directory = os.environ["HOME"]
    os.makedirs(home_directory + "/.jaxNRSur", exist_ok=True)
    try:
        print("Try loading file from cache")
        data = h5py.File(home_directory + "/.jaxNRSur/" + local_filename, "r")
        print("Cache found and loading data")
    except FileNotFoundError:
        print("Cache not found, downloading from Zenodo")
        downloaded = download_from_zenodo(
            url,
            home_directory + "/.jaxNRSur/" + local_filename,
        )
        if downloaded:
            print("Download successful, loading data")
            data = h5py.File(home_directory + "/.jaxNRSur/" + local_filename, "r")
        else:
            raise KeyError("Cannot download data from zenodo")
    return data


def h5Group_to_dict(h5_group: h5py.Group) -> dict:
    result = {}
    for key in h5_group.keys():
        local_data = h5_group[key]
        if isinstance(local_data, h5py.Dataset):
            result[key] = local_data[()]
        elif isinstance(local_data, h5py.Group):
            result[key] = h5Group_to_dict(local_data)
        else:
            raise ValueError("Unknown data type")
    return result
