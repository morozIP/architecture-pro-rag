```puml
@startuml
actor "Admin/System" as admin
participant "Local Storage (knowledge_base/)" as docs
participant "Update Script (Python)" as script
participant "Embedding Model" as ml
database "FAISS Index" as db
participant "Log File (update_log.txt)" as logs

admin -> script : Trigger (Cron/Schedule)
script -> docs : Scan for new files
docs --> script : Return .txt files
script -> ml : Generate Embeddings
ml --> script : Return Vectors
script -> db : Rebuild & Save Index
script -> logs : Write Status & Stats
@enduml
```