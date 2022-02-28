import streamlit as st
import requests


def request_prediction(model_uri, data):
    response = requests.get(url=model_uri, params={"data": data})

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(
                response.status_code, response.text))

    return response.json()


def main():
    API_URI = 'http://backend:8000/prediction'

    st.title('Automatic tag assignment')

    post = st.text_input('Contenu du post')

    predict_btn = st.button('Prédire')
    if predict_btn:
        pred = request_prediction(API_URI, post)
        st.write(f'Les tags associés à ce post sont {pred["Tags"][0]}')


if __name__ == '__main__':
    main()
