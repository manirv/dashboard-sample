import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from textwrap import dedent as d

import regression
import gunculture
###***Import***###


# Here is the Dash App Layout and interaction behaviour defined.
app = dash.Dash()
server = app.server

#Methods for header etc

# includes page/full view
def get_logo():
    logo = html.Div([

        html.Div([
            html.Img(src='http://www.colaberry.com/wp-content/uploads/2017/09/cropped-u6-1.png', height='40', width='160')
        ], className="ten columns padded")

 
    ], className="row gs-header")
    return logo


def get_header():
    header = html.Div([

        html.Div([
            html.H5(
                'Colaberry Data Doc')
        ], className="twelve columns padded")

    ], className="row gs-header gs-text-header")
    return header


def get_menu():
    menu = html.Div([

        dcc.Link('Overview   ', href='/overview', className="tab first"),
        dcc.Link('Regression   ', href='/regression', className='tab'),
        dcc.Link('Gunculture   ', href='/gunculture', className='tab'),
        ###***Menu***###

    ], className="row ")
    return menu


#Layouts
## Page layouts
overview = html.Div([  # page 1

        html.Div([


            # Row 3

            html.Div([

                html.Div([
                    html.H6('Colaberry Data Doc Overview page',
                            className="gs-header gs-text-header padded")]),
            html.Div([
                dcc.Markdown(d("""
                    

                    Here is the introduction text for Colaberry's Data Docs and what they are all about.
                """))
            ], className='three columns')

                    ])
            ])
            ])


noPage = html.Div([  # 404

    html.P(["404 Page not found"])

    ], className="no-page")

# Describe the layout, or the UI, of the app
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
            # Header
            get_logo(),
            get_header(),
            html.Br([]),
            get_menu(),
            html.Div(id='page-content')
])



# Update page
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/' or pathname == '/overview':
        return overview
    elif pathname == '/regression':
        return regression.layout
    elif pathname == '/gunculture':
        return gunculture.layout
    ###***Path***###
    else:
        return noPage    
external_css = [
#"https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
#                "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
#                "//fonts.googleapis.com/css?family=Raleway:400,300,600",
                "https://codepen.io/bcd/pen/KQrXdb.css",
                "https://raw.githubusercontent.com/colaberry/datadocgen/master/datadoc.css"
																															#                "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"
]

for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ["https://code.jquery.com/jquery-3.2.1.min.js",
               "https://codepen.io/bcd/pen/YaXojL.js"]

if __name__ == '__main__':
    app.run_server(debug=True)
