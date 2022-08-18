$Env:driver_store=$(ls $($($(Get-WmiObject Win32_VideoController).InstalledDisplayDrivers | sort -Unique).ToString().Split(',')| sort -Unique).ToString().Replace("\DriverStore\", "\HostDriverStore\")).Directory.FullName

cp "$Env:driver_store\nvcuda64.dll" C:\Windows\System32\nvcuda.dll
cp "$Env:driver_store\nvapi64.dll" C:\Windows\System32\nvapi64.dll

