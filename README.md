# Fake Tweet Detector (Binary Classification Using neural networks)
This project is a machine learning based solution to identify fake news tweets. It uses the state-of-the-art BERT (Bidirectional Encoder Representations from Transformers) model for sequence classification. The model has been trained on a large dataset of tweets to differentiate between real and fake news tweets. The model preprocesses the input tweet by encoding it into numerical values and uses an attention mechanism to give more weight to the important parts of the tweet while making predictions. The predictions made by the model are in the form of a probability score and a class label (real or fake). The deployment of this project allows for easy integration into various applications and platforms to detect fake news tweets in real-time.

## Why Neural Deep Learning is better for binary classification? 
Deep learning neural networks have several advantages for binary classification tasks compared to traditional machine learning algorithms. These advantages include:

* Ability to handle large amounts of data: Neural networks can handle large amounts of data, allowing them to learn from more examples and improve accuracy.

* Non-linear modeling: Neural networks can model complex, non-linear relationships between inputs and outputs, making them suitable for binary classification tasks that may have non-linear decision boundaries.

* Automated feature extraction: Neural networks can automatically extract relevant features from raw input data, reducing the need for manual feature engineering.

* **Robustness:** Neural networks are relatively robust to noise and outliers in the data, as they average over multiple examples.

* **Scalability:** Neural networks can be scaled up to large datasets and high dimensional input spaces by adding more layers and neurons.

Overall, deep learning neural networks offer a powerful and flexible approach to binary classification, making them a popular choice for many tasks.

## Tech Stack

**Front End:** HTML,CSS,Java

**Server:** Python

## Requirements
To run the project, you'll need the following libraries:

- PyTorch
- Transformers
- FastAPI
- Pydantic
- JSON
- Torch
- cuda
- nlkt
- uvicorn (For runing the server you can also use different services)

 
## Training The model

Just run [Main.py](https://github.com/dryruffian/Fake-Tweet-Detector-Binary-Classification-Using-neural-networks-/blob/main/main.py) for Training the model

## Deployment

To deploy this project 

* run cmd and there open the folder use save your files to


```bash
   C:\Users\username>cd {here paste the path of the file}
```

* Run the server(i am runnig it using uvicorn)
```bash
    C:\{path to you file}>uvicorn test:app
```

* Then you can just run [Twitter.htm](https://github.com/dryruffian/Fake-Tweet-Detector-Binary-Classification-Using-neural-networks-/blob/main/UI/Twitter.html)

## UI
![image](https://user-images.githubusercontent.com/88555779/215589936-4c4df859-7ff9-4303-9e44-568de17a5a4f.png)
this is a HTML and javascript Based UI for the deployment of the project

## A side Note
this is a self project and my first Project of this scale if you are see this please help me to better this project.
