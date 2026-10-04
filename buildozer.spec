[app]
title = PPLineCaller
package.name = pplinecaller
package.domain = org.example
source.dir = ./app
source.include_exts = py
version = 0.1
requirements = python3,kivy,numpy,opencv,plyer
orientation = landscape
fullscreen = 0
android.permissions = CAMERA,INTERNET
android.api = 34
android.minapi = 24
android.build_tools = 34.0.0
android.ndk = 25b
android.archs = arm64-v8a
android.accept_sdk_license = True
p4a.branch = develop
[buildozer]
log_level = 2
