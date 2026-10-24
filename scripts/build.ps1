param([Parameter(Mandatory=$true)][string]$ToolchainRoot, [switch]$DebugBuild)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskToolchain = (Resolve-Path -LiteralPath $ToolchainRoot).Path
if (-not (Test-Path -LiteralPath (Join-Path $taskToolchain 'bin\g++.exe'))) {
    throw 'ToolchainRoot must contain bin/g++.exe (w64devkit / MinGW-w64).'
}
Push-Location -LiteralPath $taskRoot
$taskOldPath = $env:PATH
$taskOldPrefix = $env:GCC_EXEC_PREFIX
try {
    New-Item -ItemType Directory -Force -Path build | Out-Null
    $taskBaseUri = [Uri]($taskRoot.TrimEnd('\') + '\')
    $taskCompilerUri = [Uri]($taskToolchain.TrimEnd('\') + '\')
    $taskPrefix = [Uri]::UnescapeDataString($taskBaseUri.MakeRelativeUri($taskCompilerUri).ToString())
    if ($taskBaseUri.Scheme -ne $taskCompilerUri.Scheme -or $taskBaseUri.Host -ne $taskCompilerUri.Host) {
        throw 'Use CMake when source and portable toolchain cannot share relative paths.'
    }
    $taskVersions = @(Get-ChildItem -LiteralPath (Join-Path $taskToolchain 'lib\gcc\x86_64-w64-mingw32') -Directory)
    if ($taskVersions.Count -ne 1) { throw 'Expected one GCC version directory.' }
    $taskVersion = $taskVersions[0].Name
    $env:PATH = (Join-Path $taskToolchain 'bin') + ';' + $taskOldPath
    $env:GCC_EXEC_PREFIX = $taskPrefix + 'lib/gcc/'
    $taskFlags = @(('-B'+$taskPrefix+'bin/'),
        ('-B'+$taskPrefix+'lib/gcc/x86_64-w64-mingw32/'+$taskVersion+'/'),
        ('-B'+$taskPrefix+'x86_64-w64-mingw32/lib/'),
        ('-L'+$taskPrefix+'lib/gcc/x86_64-w64-mingw32/'+$taskVersion+'/'),
        ('-L'+$taskPrefix+'x86_64-w64-mingw32/lib/'),
        '-std=c++17','-static','-pthread','-Wall','-Wextra','-Wpedantic','-ffp-contract=off','-Iinclude')
    if ($DebugBuild) { $taskFlags += @('-O0','-g','-D_GLIBCXX_ASSERTIONS') }
    else { $taskFlags += @('-O3','-DNDEBUG') }
    $taskCompiler = Join-Path $taskToolchain 'bin\g++.exe'
    $taskCore = @('src/search.cpp','src/io.cpp','src/batch.cpp')
    & $taskCompiler @taskFlags @taskCore 'tests/test_search.cpp' 'tests/test_io_batch.cpp' 'tests/test_batch.cpp' '-o' 'build/exactsearch_tests.exe'
    if ($LASTEXITCODE -ne 0) { throw 'Test build failed' }
    foreach ($taskPair in @(@('exactsearch','main'),@('exactsearch_profile','profile'),@('exactsearch_batch','batch_bench'))) {
        & $taskCompiler @taskFlags @taskCore ('src/'+$taskPair[1]+'.cpp') 'src/entry.cpp' '-municode' '-o' ('build/'+$taskPair[0]+'.exe')
        if ($LASTEXITCODE -ne 0) { throw ('Build failed: '+$taskPair[0]) }
    }
    Write-Output 'Built four executables in build/.'
} finally {
    $env:PATH = $taskOldPath
    $env:GCC_EXEC_PREFIX = $taskOldPrefix
    Pop-Location
}
