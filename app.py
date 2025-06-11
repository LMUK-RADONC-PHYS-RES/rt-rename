from dash import Dash, html, dcc, callback, Input, Output, State, set_props, DiskcacheManager, no_update
from dash import dash_table
import dash_daq as daq
import dash_ag_grid as dag
from utils import run_llm, run_model, parse_prompt_v2, load_guideline, check_TG263_name, parse_prompt, parse_filenames, read_guideline, parse_csv, parse_dicom, update_dicom
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import datetime
import io
import json
import diskcache
import pydicom
import tempfile
from time import sleep

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)
file_content = {}

app = Dash(name ='rt-rename', title="RT-Rename", external_stylesheets=[dbc.themes.UNITED])

columnDefs = [
    {"field": "local name"},
    {"field": "TG263 name"},
    {"field": "confidence"},
    {
        "field": "verify",
        "cellStyle": {
            "styleConditions": [
                {
                    "condition": "params.value == 'pass'",
                    "style": {"color": "green"},
                },
                {
                    "condition": "params.value == 'fail'",
                    "style": {"color": "red"},
                },
            ],
        },
    },
    {"field": "accept",
     "flex": 1,
     "width":'10%',
     'editable': True},
    {"field": "comment",
     "flex": 1,
     'editable': True},
    {"field": "raw output",
     "flex": 1,
     'editable': True},
]

grid = dag.AgGrid(
    id="main-data-table",
    rowData=[],
    columnDefs=columnDefs,
    dashGridOptions={"domLayout": "autoHeight"},
    style={"height": None},
    csvExportParams={"fileName": "ag_grid_test.csv"},
)

app.layout =  html.Div(
    [
        html.Div(
            className="column",
            children=[
                html.H3(
                    children="RT-Rename v0.2",
                    style={
                        "textAlign": "left",
                        "margin": "1%",
                    },
                ),
            ],
            style={
                "width": "70%",
                "display": "inline-block",
                "verticalAlign": "bottom",
            },
        ),
        html.Div(
            className="column",
            children=[html.P("Status:", id="status-static")],
            style={
                "width": "10%",
                "display": "inline-block",
                "verticalAlign": "bottom",
                "textAlign": "right",
            },
        ),
        html.Div(
            className="column",
            children=[
                html.P(
                    "idle",
                    id="status-bar",
                    style={"color": "green", "margin-left": "10px"},
                )
            ],
            style={
                "width": "20%",
                "display": "inline-block",
                "verticalAlign": "bottom",
            },
        ),
        html.Hr(),
        html.H4(
            children="Settings",
            style={
                "textAlign": "left",
                "margin": "1%",
            },
        ),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="column",
                    children=[
                        html.P(
                            "Select a nomenclature:",
                            style={"verticalAlign": "top"},
                        ),
                        dcc.Dropdown(
                            ["TG263", "TG263_reverse"],
                            "TG263",
                            multi=False,
                            id="guideline",
                            style={"width": "80%"},
                        ),
                        html.P(
                            "Select which regions to include:",
                            style={
                                "verticalAlign": "top",
                                "margin-top": "10px",
                            },
                        ),
                        dcc.Dropdown(
                            [
                                "Thorax",
                                "Head and Neck",
                                "Abdomen",
                                "Limb",
                                "Pelvis",
                                "Body",
                                "Limbs",
                            ],
                            ["Thorax", "Head and Neck", "Body"],
                            multi=True,
                            id="regions",
                            style={"width": "80%"},
                        ),
                    ],
                    style={
                        "width": "33%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
                html.Div(
                    className="column",
                    children=[
                        html.P(
                            "Remove target volumes?",
                            style={"verticalAlign": "top"},
                        ),
                        dcc.Dropdown(
                            ["False", "True"],
                            "False",
                            multi=False,
                            id="TV-filter",
                            style={"width": "80%"},
                        ),
                        html.P(
                            "Structure Set",
                            style={
                                "verticalAlign": "center",
                                "margin-top": "10px",
                            },
                        ),
                        dcc.Upload(
                            id="upload-data",
                            children=html.Div(
                                [
                                    html.Br(),
                                    "Drag and Drop or click to select Files",
                                    html.Br(),
                                    html.Br(),
                                ]
                            ),
                            style={
                                "width": "80%",
                                "lineHeight": "20px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "textAlign": "center",
                                "margin": "10px",
                                "verticalAlign": "center",
                            },
                            multiple=True,
                        ),
                    ],
                    style={
                        "width": "33%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
                html.Div(
                    className="column",
                    children=[
                        html.P("Model:", style={"verticalAlign": "top"}),
                        dcc.Dropdown(
                            [
                                "Llama 3.1:70B",
                                "Llama 3.3:70B",
                                "Qwen 2.5",
                                "Llama 3.2",
                                "qwq",
                                "R1",
                                "V3-cloud",
                                "R1-cloud",
                                "L3-cloud",
                                "L3R-cloud"
                            ],
                            "Llama 3.1:70B",
                            multi=False,
                            id="model",
                            style={"width": "80%"},
                        ),
                        html.P(
                            "Prompt:",
                            style={
                                "verticalAlign": "top",
                                "margin-top": "10px",
                            },
                        ),
                        dcc.Dropdown(
                            ["v1", "v2", "v3", "v4", "v5","v6"],
                            "v1",
                            multi=False,
                            id="prompt",
                            style={"width": "80%"},
                        ),
                    ],
                    style={
                        "width": "33%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
            ],
            style={
                "marginLeft": "1%",
                "marginRight": "1%",
            },
        ),
        html.Hr(),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="column",
                    children=[
                        html.Button(
                            "Start renaming",
                            id="button-run-model",
                            n_clicks=0,
                            style={
                                "borderRadius": "5px",
                                "width": "80%",
                                "background-color": "#32a852",
                                "margin-top": "15px",
                            },
                        )
                    ],
                    style={"width": "33%", "display": "inline-block"},
                ),
                html.Div(
                    className="column",
                    children=[html.Div(
                        children=[
                            dcc.Input(
                                placeholder="Enter patient name/ID here:", 
                                type="text", 
                                id="patient-name",
                                style={
                                    "margin-top": "15px",
                                    "text-allign":"center", 
                                    "width":"87%"}),])],
                    style={
                        "width": "33%",
                        "display": "inline-block",
                        "text-allign":"center",
                    },
                ),
                html.Div(
                    className="column",
                    children=[
                        html.Button(
                            "Export csv",
                            id="button-export",
                            n_clicks=0,
                            style={
                                "borderRadius": "5px",
                                "width": "100%",
                                "background-color": "#33b0f2",
                                "margin-top": "15px",
                                "float": "right",
                            },
                        )
                    ],
                    style={"width": "16.5%", "display": "inline-block"},
                ),
                html.Div(
                    className="column",
                    children=[
                        html.Button(
                            "Export RTstruct",
                            id="button-export-dcm",
                            n_clicks=0,
                            style={
                                "borderRadius": "5px",
                                "width": "100%",
                                "background-color": "#33b0f2",
                                "margin-top": "15px",
                                "float": "right",
                            },
                        )
                    ],
                    style={"width": "16.5%", "display": "inline-block"},
                ),
                dcc.Download(id="download-dataframe-dcm")
            ],
            style={
                "marginLeft": "10px",
                "marginRight": "10px",
            },
        ),
        html.Div(children=[grid], style={"margin": "1%"}),
    ],
    style={"margin": "1%"},
)

# Load data from filenames
@callback(
    Output('main-data-table', 'rowData'),
    Input('upload-data', 'filename'),
    Input('upload-data', 'contents'),
    State('TV-filter', 'value'),
    prevent_initial_call=True
)
def update_on_file_load(filename,contents,filter_tv):
    print(f'Filtering target volumes: {filter_tv}')
    if 'csv' in filename[0]:
        data = parse_csv(contents[0], filename[0])
        set_props('status-bar', {'children': html.P(f"{len(data)} structures loaded from {filename[0]}")})
        return data
    
    elif '.DCM' in filename[0] or '.dcm' in filename[0]:
        if contents is None:
            return ""
        
        # read structure names from DICOM file into dataframe for GUI
        data = parse_dicom(contents[0], filename[0], filter_tv)
        set_props('status-bar', {'children': html.P(f"{len(data)} structures loaded from {filename[0]}")})
        
        # read entire DICOM into buffer for export later
        decoded = base64.b64decode(contents[0].split(',')[1])
        
        # Read DICOM file from bytes
        dicom_file = pydicom.dcmread(io.BytesIO(decoded))
        file_content['pydicom'] = dicom_file
        file_content['filename'] = filename[0]
        
        return data
    
    else:
        data = parse_filenames(filename,filter_tv)
        set_props('status-bar', {'children': html.P(f"{len(data)} structures loaded")})
        return data


# Run model
@callback(
    Input('button-run-model', 'n_clicks'),
    State('guideline', 'value'),
    State('regions', 'value'),
    State('model', 'value'),
    State('prompt', 'value'),
    State('main-data-table', 'rowData'),
    State("main-data-table", "columnDefs"),
    running=[
        (Output("button-run-model", "disabled"), True, False),
    ],
    background=True,
    manager=background_callback_manager,
    prevent_initial_call=True
)
def update_on_model_run(n_clicks, guideline, regions, model, prompt, data, columnDefs):
    updated_data = run_model(model,prompt,guideline,regions,data,columnDefs,uncertain=False)
    print(updated_data)
    set_props('main-data-table', {'rowData': updated_data})
    set_props('status-bar', {'children': html.P("Model run finished!")})

# export data as csv
@callback(
    Output("main-data-table", "exportDataAsCsv"),
    Output("main-data-table", "csvExportParams"),
    Input("button-export", "n_clicks"),
    State("patient-name", "value"),
    prevent_initial_call=True,
)
def export_data_as_csv(n_clicks,patient_name):
    print('button clicked')
    return True, {"fileName": f"{patient_name}.csv"}

@app.callback(
    Output("download-dataframe-dcm", "data"),
    Input("button-export-dcm", "n_clicks"),
    State("main-data-table", "rowData"),
    prevent_initial_call=True
)
def download_file(n_clicks, rowData):
    if 'pydicom' in file_content:
        # Create a temp file to serve as a download
        temp = tempfile.NamedTemporaryFile(delete=False, suffix='.dcm')
        # Update the structure names in the DICOM file
        file_content['pydicom'] = update_dicom(file_content['pydicom'], rowData)
        pydicom.dcmwrite(temp.name, file_content['pydicom'])
        temp.close()
        filename_new = file_content['filename'].strip('.dcm').strip('.DCM') + '_renamed.dcm'
        return dcc.send_file(path = temp.name, filename=filename_new)
    return no_update

@callback(
    Input("main-data-table", "cellRendererData"),
    State("main-data-table", "rowData"),
    prevent_initial_call=True,
)
def accept_structures(n, data):
    structure = data[int(n["rowId"])]
    data[int(n["rowId"])]["accept"] = n["value"]
    set_props('main-data-table', {'rowData': data})


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port='8055')
