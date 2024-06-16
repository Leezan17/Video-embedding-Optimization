# Video-embedding-Optimization

This repository contains the code for a deep video embedding optimization project that aims to improve video retrieval performance by directly optimizing for top-k precision.

## Dataset

The project uses the UCF101 dataset, which can be downloaded from [UCF101 Dataset](http://crcv.ucf.edu/data/UCF101.php).

## Installation

### Prerequisites

- Python 3.6+
- PyTorch 1.7.0+
- torchvision 0.8.0+
- Other dependencies can be installed via `requirements.txt`

### Steps

1. **Clone the repository:**

    ```bash
    git clone https://github.com/Leezan17/Video-embedding-Optimization.git
    cd Video-embedding-Optimization
    ```

2. **Set up a virtual environment (optional but recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download and extract the UCF101 dataset:**

    Download the dataset from [UCF101 Dataset](http://crcv.ucf.edu/data/UCF101.php) and extract it to a folder named `UCF-101` in the root directory of this project.

5. **Organize the dataset:**

    Run the `organize_ucf101.py` script to prepare the dataset:

    ```bash
    python organize_ucf101.py
    ```

6. **Training the Model:**

    To start training the model, run the `train_model.py` script:

    ```bash
    python train_model.py
    ```

## Project Structure

- `organize_ucf101.py`: Script to organize the UCF101 dataset into train and validation sets.
- `train_model.py`: Script to train the video embedding model.
- `ucf101_dataset.py`: Custom PyTorch dataset class for loading UCF101 videos.
- `video_model.py`: Contains the model definition for the video embedding network.

## Additional Information

For detailed explanations of the code and methodologies used in this project, please refer to the accompanying research paper included in this repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please contact Leeza Nadeem at [lnadeem@usfca.dons.edu](mailto:lnadeem@usfca.dons.edu).

