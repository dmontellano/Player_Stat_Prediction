# NFL Player Stat Prediction

This project predicts NFL stats for quarterbacks (QBs), running backs (RBs), and wide receivers (WRs) using polynomial linear regression models. It uses historical play-by-play data to calculate features from previous seasons and trains models to predict touchdowns.

## Features
- Polynomial regression for QBs, RBs, and WRs
- Loads and processes NFL play-by-play data
- Models evaluate performance using RMSE and R²
- Highlights most accurate player predictions

## Installation
1. Clone this repository
   ```bash
   git clone https://github.com/dmontellano/Player_Stat_Prediction.git
   ```
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Update the `project_dir` path in the script to your local directory where the data is stored. Then, run the script to generate predictions.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.