# Fake Tweet Detector (Binary Classification Using neural networks)
This project is a machine learning based solution to identify fake news tweets. It uses the state-of-the-art BERT (Bidirectional Encoder Representations from Transformers) model for sequence classification. The model has been trained on a large dataset of tweets to differentiate between real and fake news tweets. The model preprocesses the input tweet by encoding it into numerical values and uses an attention mechanism to give more weight to the important parts of the tweet while making predictions. The predictions made by the model are in the form of a probability score and a class label (real or fake). The deployment of this project allows for easy integration into various applications and platforms to detect fake news tweets in real-time.

# Requirements
To run the project, you'll need the following libraries:

- PyTorch
- Transformers
- FastAPI
- Pydantic
- JSON
- Torch
- cuda
- nlkt
- uvicorn(For runing the server you also use different services)


## Deployment

To deploy this project 

* run cmd and there open the folder use save your files to


```bash
   C:\Users\username>cd {here past the path of the file}
```

* Run the server(i am runnig it using uvicorn)
```bash
    C:\{path to you file}>uvicorn test:app
```

* Then you can run [Twitter.htm](https://www.github.com/octokatherine)
