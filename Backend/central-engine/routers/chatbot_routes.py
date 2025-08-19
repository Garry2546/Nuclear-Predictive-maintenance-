from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse







chatbotRouter = APIRouter()




@chatbotRouter.get("/checklist-data")
def return_checklist():
    try:
        with open("checklist.txt", "r") as file:
            checklist_data = file.readlines()  # Read all lines from the file
        return JSONResponse(content={"checklist": checklist_data})
    except FileNotFoundError:
        return JSONResponse(content={"error": "checklist.txt not found"}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)




@chatbotRouter.get("/summary-extractor")
def return_summary():
    try:
        with open("summary.txt", "r") as file:
            checklist_data = file.readlines()  # Read all lines from the file
        return JSONResponse(content={"summary": checklist_data})
    except Exception as e:
        print(e)
