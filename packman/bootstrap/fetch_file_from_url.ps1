# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

param(
[Parameter(Mandatory=$true)][string]$sourceUrl=$null,
[Parameter(Mandatory=$true)][string]$output=$null
)
$source = $sourceUrl
$filename = $output

$req = [System.Net.httpwebrequest]::Create($source)
$req.cookiecontainer = New-Object System.net.CookieContainer

Write-Host "Connecting to $source ..."
$res = $req.GetResponse()

if($res.StatusCode -eq "OK") {
  Write-Host "Downloading ..."
  [int]$goal = $res.ContentLength
  $reader = $res.GetResponseStream()
  $writer = new-object System.IO.FileStream $fileName, "Create"
  [byte[]]$buffer = new-object byte[] 4096
  [int]$total = [int]$count = 0
  do
  {
    $count = $reader.Read($buffer, 0, $buffer.Length);
    $writer.Write($buffer, 0, $count);
    $total += $count
    if($goal -gt 0) {
        Write-Progress "Downloading $url" "Saving $total of $goal" -id 0 -percentComplete (($total/$goal)*100)
    } else {
        Write-Progress "Downloading $url" "Saving $total bytes..." -id 0
    }
  } while ($count -gt 0)
 
  $reader.Close()
  $writer.Flush()
  $writer.Close()
}

