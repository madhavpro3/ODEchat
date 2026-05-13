# ODEchat

## ODEchat is an open-source integrated development environment for building and running PK/PD, Quantitative Systems Pharmacology Models. 

### Some of its features 
* **Track all the model changes** and allows you to go back to a prior version
* Run the analysis in a chat style window so that you can **trace your results**
* **Ensures your analysis is reproducible** so you can focus only on the its interpretation
* Add **in-place documentation** on assumptions and modeling choices so that you cant miss  

A demo version of the tool can be accessed at [Free Streamlit cloud](https://odechat.streamlit.app/)

## Installation instructions (Self-hosting)

### Through Docker Image (Simple setup)
* Install [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) from the official website
* Start Docker Desktop
* Open cmd prompt and type - `docker pull madhavpro3/odechat:latest`
* Type: `docker run -d -p 8501:8501 madhavpro3/odechat:latest`
* Application will be running at localhost:8501

## Examples
Following examples are in the examples folder
* [Cao Y, Balthasar JP, Jusko WJ. Second-generation minimal physiologically-based pharmacokinetic model for monoclonal antibodies. J Pharmacokinet Pharmacodyn. 2013 Oct;40(5):597-607. doi: 10.1007/s10928-013-9332-2](https://github.com/madhavpro3/ODEchat/blob/main/examples/mPBPK_Cao_Balthasar_Jusko.md) 

***

## Capabilities (v0.1)
* Simulation
* Plotting
* Cailbration
* Updating parameters -> Creates a new model state
* Re-visiting prior model states
* Saving the analysis as a chat
