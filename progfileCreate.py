import os
with open(os.path.join('C:\\Users\\Baris\\Desktop\\StreamLit\\Proje 1','Procfile'), "w") as file1:
    toFile = 'web: sh setup.sh && streamlit run app.py'
    
file1.write(toFile)
