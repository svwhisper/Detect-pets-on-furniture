# Pet Monitoring Camera

Ever wonder if your dog or cat is sneaking onto the furniture while you're away? This project uses a TP-Link camera broadcasting an RTSP feed, [YOLO real-time object detector](https://docs.ultralytics.com/models/yolo11/#overview) to catch couch-surfing pets in the act. Monitor their mischief and win the battle of "No pets on the furniture!"

This code was originally created by Tanner Allen, but has been substantially replaced with the help of Google Gemini 3 Pro.  It was originally intended for use on a Raspberry pi, but I didn't want a pi dedicated to this, so have adjusted it to run on an Apple Mac Mini, using the Apple Neural Engine (ANE) to offload inference from the CPU.  As a result, the CPU load is very low, even with the large Yolo model.  Use a smaller model if you like.

I have not tested this on other platforms, but believe it will work, as there is logic to cater for that.

The original project triggered a wireless dog collar, but I've removed that code and use it only for detection.  I have a buzzer on order and will trigger that using Node Red, which receives the mqtt messages published by this project.

## Getting Started
1. Clone this repository.
2. Update the `config.json` file with your RTSP feed URL and tweak any other parameters you want.

## Usage
1. Run the `animal_monitor.py` script to launch the RTSP stream and start the object detector.  On my Mac, I use PM2 to daemonise and manage this process.
2. If a dog or cat is detected on the furniture, the program will save the image locally, ftp it to my NAS and publish a detected message (with saved image filename on the NAS) to mqtt.  In my case, I have code in Node Red that subscribes to the topic and uses Pushover to send an alert to my mobile phone with a copy of the image.

## License
This project is licensed under the MIT License.

*Created by Tanner J. Allen 2025. See [tannerjallen.com](https://tannerjallen.com).*
*Modified by Dave Wilson 2025.