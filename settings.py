
def css():
    custom_css = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header .decoration {
        display: flex;
        align-items: center;
    }
    header .decoration .stApp {
        display: none;
    }
    header .decoration:before {
        content: '';
        display: inline-block;
        width: 40px;
        height: 40px;
        background: url('https://www.itera.ac.id/wp-content/uploads/2016/02/logo-itera-oke.jpg') no-repeat center center;
        background-size: contain;
        margin-right: 10px;
    }
    </style>
    """
    return custom_css