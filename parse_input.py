import re
from typing import List, Dict, Callable
from pydantic import BaseModel
import json
import pandas as pd
from scipy import integrate

#------------------- Pydantic data classes -------------------

class BasicResponse(BaseModel):
    thinking: str
    response: str

class SimParameters(BaseModel):
    dose:float = 10
    doseunits:str = 'mpk'
    doseregimen:str = ''
    time:float = 21
    timeunits:str = 'day'

class UpdateParameters(BaseModel):
    parametername: str=''
    value: float=0

class PlotParameters(BaseModel):
    X: str
    Y: str
    Yscale_log: bool = False
    Xscale_log: bool = False 

#------------------------------------
def llm_call(prompt,outputformat):
    payload = {
    "model": "gemma3:4b", # your installed model name
    "messages": [{"role":"system","content":prompt}],
    "stream": False,
    "format":outputformat.model_json_schema(),
    "temperature":0.
    }
    response = requests.post("http://localhost:11434/api/chat", json=payload)

    if response.status_code == 200:
        data = response.json()
        if "message" in data and "content" in data["message"]:
            reply = data["message"]["content"]
        else:
            reply = "Got response, but no message content was found."
    else:
        reply = f"Ollama returned status code {response.status_code}"
        
    return reply

def extract_xml(text: str, tag: str) -> str:
    """
    Extracts the content of the specified XML tag from the given text. Used for parsing structured responses 

    Args:
        text (str): The text containing the XML.
        tag (str): The XML tag to extract content from.

    Returns:
        str: The content of the specified XML tag, or an empty string if the tag is not found.
    """
    match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return match.group(1) if match else ""

def findaction(userinput: str, routes: Dict[str,List]):
    user_input_split=userinput.lower().split(" ")
    bestroute,bestscore="",-1

    for key,val in routes.items():
        curscore=0
        for w in user_input_split:
            for rw in val[0]:
                if w==rw:
                    curscore+=1
                elif w.find(rw)>=0:# this is useful if input is "workbook1" instead of "workbook 1"
                    curscore+=0.5

        if curscore>bestscore:
            bestscore=curscore
            bestroute=key

    return {"thinking":"simple","response":bestroute}

    # """Route input to specialized prompt using content classification."""
    # # First determine appropriate route using LLM with chain-of-thought

    # # break the words in the actions space
    # # break the words in the user ask
    # # find the similarity score = # of common words in user ask and actions space
    # # Note: Sub-string search wont work because of the plurals
    # scores=[0 for i in range(len(routes))]
    # userinput_split=userinput.lower().split(" ")
    # bestroute_inx,bestscore=0,0
    # for i in range(len(routes)):
    #     route_split=routes[i].split(" ")
    #     for w in user_input_split:
    #         if w in route_split:
    #             scores[i]+=1
    #     if scores[i]>bestscore:
    #         bestscore=scores[i]
    #         bestroute_inx=i

    # return {"thinking":"simple","response":routes[bestroute_inx]}

    # for r in routes:
    #     if r in userinput.lower():
    #         return {"thinking":"simple","response":r}

    # selector_prompt = f"""
    # Analyze the input and select the most appropriate route from these options: {routes}
    # You have to definitely pick one route. 
    # provide your thinking and the response in JSON format:
    # Input: {userinput}""".strip()
    
    # route_response = llm_call(selector_prompt,BasicResponse)

    # return json.loads(route_response)
    
def extract_simparameters(input: str):
    w=input.split(" ")
    if len(w)<4:
        return SimParameters(dose=0,doseunits="mpk",doseregimen='',time=0,timeunits="days") 
    return SimParameters(dose=extract_num(w[1]),doseunits="mpk",doseregimen=w[2],time=extract_num(w[3]),timeunits="days")
    # return {"dose":extract_num(w[1]),"doseunits":"mpk",
    #     "doseregimen":w[2],"time":extract_num(w[3]),"timeunits":"days"}
    # Dose, time, dose regimen
    # user only provides 1 scenario. not like 3 & 5mpk or qw and q2w. These can be given with a ';' separation

    # extractor_prompt=f"""Extract dose, doseunits, time, and timeunits from the given text in JSON format.
    
    # Descriptions for these are: 
    # dose:dose value without units.
    # doseunits: dose units.
    # doseregimen: dose regimen.
    # time:simulation time without units.
    # timeunits: simulation time units.

    # Some examples are
    # ---
    # USER: simulate the model for 5mpk at q2w for 63days
    # RESULT:
    # dose:5
    # doseunits:mpk
    # doseregimen:q2w
    # time:63
    # timeunits:days

    # ---
    # USER: run the model for 3 mg
    # RESULT:
    # dose:3
    # doseunits:mg
    # ---
    # USER: simulate the model for qw 5 MPK for 21 hours
    # RESULT:
    # dose:5
    # doseunits:MPK
    # doseregimen:qw
    # time:21
    # timeunits:hours
    # ---
    # USER: simulate the model
    # RESULT:

    # Input: {input}""".strip()
    # sim_p=SimParameters.model_validate_json(llm_call(extractor_prompt,SimParameters))

    # return sim_p

def extract_updateparameters(input: str):
    w=input.split(" ")
    up_p=UpdateParameters(parametername=w[1],value=w[3])
    return up_p
    # match = re.search(f'update (.*?) to (.*?)', input.lower(), re.DOTALL)
    # if match:
    #     up_p=UpdateParameters()
    #     up_p.parametername=match.group(1)
    #     up_p.value=float(match.group(2))

    #     return up_p

    extractor_prompt=f"""Extract parameter name and its value from the given text in JSON format.
    
    Descriptions for these are: 
    parametername: parameter name.
    value: value of the parameter.

    Some examples are
    ---
    USER: update Vc to 10
    RESULT:
    parametername:Vc
    value:10
    ---
    USER: update vc to 2
    RESULT:
    parametername:vc
    value:2
    ---
    USER: update cl to 13
    RESULT:
    parametername:cl
    value:13
    ---

    Input: {input}""".strip()
    up_p=UpdateParameters.model_validate_json(llm_call(extractor_prompt,UpdateParameters))

    return up_p


def extract_plotparameters(input: str):
    # Example: plot x,xscale and y,yscale
    # checks
    # 1) Check if 1 or 2 inputs are provided
    # 2) check if the inputs are among the model species
    w=input.split(" ")

    if len(w)<4:
        return PlotParameters(X='',Y='')
    return PlotParameters(X=w[1].split(",")[0],Xscale_log=int(w[1].split(",")[1]),Y=w[3],Yscale_log=int(w[3].split(",")[1]))

    # extractor_prompt=f"""Extract parameter names from the given text in JSON format.
    
    # Some examples are
    # ---
    # USER: plot a and b
    # RESULT:
    # X:a
    # Y:b
    # ---
    # USER: plot w and f
    # RESULT:
    # X: w
    # Y: f
    # ---

    # Input: {input}""".strip()
    # plot_p=PlotParameters.model_validate_json(llm_call(extractor_prompt,PlotParameters))

    # return plot_p

def extract_metricname(input:str,simdata,t=0) -> str:
    metrics_available={"cmax","auc","rolast","roattime"}
    metric_name=input.split("find")[1].strip()
    if metric_name in metrics_available:
        return metric_name

    return ""

def extract_workbooknum(input:str) -> int:
    match = re.search(r'\d+',input)
    print(match.group())
    return int(match.group())

def extract_modelnum(input:str) -> int:
    match = re.search(r'\d+',input)
    print(match.group())
    return int(match.group())

def extract_num(input:str) -> float:
    match = re.search(r'\d+',input)
    return float(match.group())
