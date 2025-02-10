# face-detection-ai

1. Install venv
python3 -m venv venv
source venv/bin/activate
pip install tensorflow

2. Insert source venv
source venv/bin/activate

3. Install requirements
pip install -r requirements.txt 
if you want to install dependencies

4. To run training
python3 main-training.py

BOILER PLATE
Right now
/app
    /models
    /routes
    /static
    /templates
/dataset
    /test
        /angry
        /happy
        /neutral
        /sad
        /suprise
    /train
        /angry
        /happy
        /neutral
        /sad
        /suprise
/venv
main-training.py
requirements.txt
stress_detection_model.h5


expected
fastapi-dashboard/
│── app/
│   ├── templates/
│   │   ├── index.html
│   │   ├── result.html
│   ├── static/
│   │   ├── styles.css
│   ├── models/
│   │   ├── cnn_model.py      # File model CNN
│   ├── routes/
│   │   ├── dashboard.py      # Route untuk dashboard
│   │   ├── api.py            # Route untuk API komunikasi
│   ├── services/
│   │   ├── ml_runner.py      # Logic untuk menjalankan model
│   ├── main.py
│── data/                     # Dataset untuk training/test
│── requirements.txt
│── README.md
