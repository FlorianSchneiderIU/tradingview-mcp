$aumid = 'TradingView.Desktop_n534cwy3pjxzj!TradingView.Desktop'

if (-not ('TVDebug.Launcher' -as [type])) {
    Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

namespace TVDebug
{
    [ComImport]
    [Guid("2E941141-7F97-4756-BA1D-9DECDE894A3D")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IApplicationActivationManager
    {
        int ActivateApplication(
            [MarshalAs(UnmanagedType.LPWStr)] string appUserModelId,
            [MarshalAs(UnmanagedType.LPWStr)] string arguments,
            uint options,
            out uint processId);
    }

    [ComImport]
    [Guid("45BA127D-10A8-46EA-8AB7-56EA9078943C")]
    class ApplicationActivationManager
    {
    }

    public static class Launcher
    {
        public static uint Activate(string aumid, string args, out int hresult)
        {
            var mgr = (IApplicationActivationManager)new ApplicationActivationManager();
            uint processId;
            hresult = mgr.ActivateApplication(aumid, args, 0, out processId);
            return processId;
        }
    }
}
"@
}

$hresult = 0
$processId = [TVDebug.Launcher]::Activate($aumid, '--remote-debugging-port=9222', [ref]$hresult)

'HR=0x{0:X8} PROCESS={1}' -f ($hresult -band 0xffffffff), $processId

if ($hresult -ne 0) {
    throw ('ActivateApplication failed: 0x{0:X8}' -f ($hresult -band 0xffffffff))
}

Start-Sleep -Seconds 2

try {
    Invoke-WebRequest 'http://127.0.0.1:9222/json/version' |
        Select-Object -ExpandProperty Content
}
catch {
    Write-Warning 'TradingView started, but http://127.0.0.1:9222/json/version did not answer.'
}