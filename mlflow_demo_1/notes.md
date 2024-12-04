set_tracking_ui()
set the default tracking URI of choice for current run
<uri>
empty string->mlruns
foldername "./<name>"
databricks workspace "databricks://profileName"

get_tracking_ui()
Get tracking URI



create_experiment()
name
artifact_location (optional)
tags
Return

set_experiment()
experiment_name
experiment_id


start_run()
run_id
experiment_id
run_name
nested


active_run() and last_active_run()