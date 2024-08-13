# Machine Learning Exercises

## Resources

- https://www.kaggle.com/learn
- https://www.harvardonline.harvard.edu/course/introduction-data-science-python

## How to docker?

Model binaries are saved to `data/models` folder.

```sh
docker build -t housing-price-prediction .
docker run \
  -v ./data:/data \
  -v ./models:/models \
  -v ./housing_price_prediction:/app/housing_price_prediction \
  housing-price-prediction train_model.py
```


## How to run streamlit apps?

```sh
cd streamlit
streamlit run <project>/app.py
```