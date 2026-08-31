Option Explicit
Dim shell, fso, root, pythonw, launcher, cmd
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
root = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = fso.BuildPath(root, ".venv\Scripts\pythonw.exe")
launcher = fso.BuildPath(root, "scripts\windows_launcher.py")
If Not fso.FileExists(pythonw) Then
  MsgBox "Runtime environment is not installed. Run RUN_WINDOWS.bat first.", 16, "Football Insight"
  WScript.Quit 1
End If
cmd = Chr(34) & pythonw & Chr(34) & " " & Chr(34) & launcher & Chr(34)
shell.CurrentDirectory = root
shell.Run cmd, 0, False
