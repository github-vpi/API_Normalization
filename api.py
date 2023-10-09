import json
from fastapi import HTTPException, FastAPI, UploadFile, File, Form
import json as json
from domain import normalize_API, normalize_curve
from typing import List
from fastapi import UploadFile, HTTPException
import lasio
import pandas as pd
import tempfile


app = FastAPI()


@app.get('/hello')
async def hello():
    return {"message": "Hello World 2"}


@app.get('/fullprocess')
async def upload_las(las_files: List[UploadFile] = File(...), path_df: UploadFile = File(...), query_params: str = Form(...)):
    dfs = []  # list to store DataFrames

    # Read the CSV file into a DataFrame
    path_df_contents = await path_df.read()  # read the file as bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(path_df_contents)
    path_df = pd.read_csv(temp_file.name)
    df_list = []
    for las_file in las_files:
        # Debug: print out some information about the las_file
        print(f"Processing file: {las_file.filename}")
        # Check if the uploaded file is a LAS file
        if las_file.filename.split('.')[-1].lower() != 'las':
            raise HTTPException(status_code=415, detail="File attached is not a LAS file")

        # Read the LAS file into a DataFrame
        # try:
        file_contents = await las_file.read()  # read the file as bytes
        print(f"File size: {len(file_contents)} bytes")  # print the size of the byte data

        # Create a temporary file on disk to store the contents
        with tempfile.NamedTemporaryFile(delete=False, suffix=".las") as temp_file:
            temp_file.write(file_contents)
        
        # Use lasio to read from the temporary file
        
        las = lasio.read(temp_file.name)
        lasdf = las.df()         
        lasdf['WELL'] = las.well.WELL.value
        lasdf['DEPTH'] = lasdf.index
        df_list.append(lasdf)
        df = pd.concat(df_list, sort=True)
            # Validate query_params using QueryParams schema
    try:
        # query_params_in_json = json.dumps(query_params)
        parameter = json.loads(query_params)
        # input_params_schema(**parameter)
    except ValueError as e:
        return {"detail": str(e)}
    except Exception as e:
        return {"detail": "Invalid JSON"}
    
    # Run prediction and return the result in JSON format
    predicted_result = normalize_API(df, path_df, parameter)
    return predicted_result

@app.get('/onlynormalization')
async def upload_las(path_df: UploadFile = File(...), query_params: str = Form(...)):
    dfs = []  # list to store DataFrames

    # Read the CSV file into a DataFrame
    path_df_contents = await path_df.read()  # read the file as bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
        temp_file.write(path_df_contents)
    path_df = pd.read_csv(temp_file.name)
    try:
        # query_params_in_json = json.dumps(query_params)
        parameter = json.loads(query_params)
        # input_params_schema(**parameter)
    except ValueError as e:
        return {"detail": str(e)}
    except Exception as e:
        return {"detail": "Invalid JSON"}
    
    # Run prediction and return the result in JSON format
    predicted_result = normalize_curve(path_df, parameter)
    return predicted_result
