copy RUN.BAT ..\
cd ..

cd ..
del release.zip
rmdir release /s /Q
tar.exe -a -c -f release.zip ECG
mkdir release
tar.exe -xf release.zip -C release
del ECG\RUN.BAT
