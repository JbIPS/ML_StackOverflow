import streamlit as st
import requests


def request_prediction(model_uri, data):
    response = requests.post(url=model_uri, json={"postContent": data})

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(
                response.status_code, response.text))

    return response.json()


def main():
    API_URI = 'http://backend:8000/prediction'

    st.set_page_config("AutoTags")
    st.title('Prédiction automatique de tags StackOverflow')

    st.text('''
    Pour tester le modèle de prédiction,
    coller un post StackOverflow dans le cadre suivant.
    ''')
    use_example = st.checkbox('Utiliser un post d\'exemple')
    post = st.text_area('Contenu du post', '''
I'm Getting Data from an external API, in which i convert to JSON then i add the data in context to use in my Template. the problem is that the different variables in the API have a specific assigned number. i want to store that number in my database so i can assign them to specific objects on my website.

here is the code

models.py =
api_text = models.CharField(max_length=100, blank=True, null=True,)

Views.py

def projectdetails(request, pk):
    url = 'XXX'
    parameters = {
        'slug': 'bitcoin',
        'convert': 'USD',
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': 'XXX'
    }
    session = Session()
    session.headers.update(headers)
    response = session.get(url, params=parameters)
    api_price = response.json()
    coin = Coin.objects.get(id=pk)
    context = {'coin':coin,
        'gimmeprice':api_price['data']['1']['quote']['USD']['price'],
    }
    return render(request, 'blog/project_details_coin.html', context)

the [1] in the context is where i want to use a variable that i assign to "api_text"

im having troubles figuring out with Django Model Query to use and how\
''' if use_example else '',
                        placeholder="Coller le contenu d'un post StackOverflow ici",
                        )
    predict_btn = st.button('Prédire')
    if predict_btn:
        pred = request_prediction(API_URI, post)
        st.write(f'Les tags associés à ce post sont {pred["Tags"][0]}')


if __name__ == '__main__':
    main()
