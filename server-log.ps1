# Set the path to the PuTTY executable
$puttyPath = "C:\Program Files\PuTTY\putty.exe"

# Set the remote server information
$remoteServer = "davinci.fmph.uniba.sk"
$remoteUser = "masny5"
$remotePassword = "xg6fxgfk6c"
Start-Process $puttyPath -ArgumentList "-ssh $remoteUser@$remoteServer -pw $remotePassword"
