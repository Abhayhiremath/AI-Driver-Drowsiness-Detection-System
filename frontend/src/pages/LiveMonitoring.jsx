import { useEffect, useState } from "react";

function LiveMonitoring() {

  const [status, setStatus] = useState("Awake");

  useEffect(() => {

    const interval = setInterval(() => {

      fetch("http://localhost:5000/status")
        .then((response) => response.json())
        .then((data) => {

          setStatus(data.status);

        })
        .catch((error) => {

          console.log("Status Fetch Error:", error);

        });

    }, 1000);

    return () => clearInterval(interval);

  }, []);

  return (

    <div className="min-h-screen bg-gray-100 flex flex-col items-center p-6">

      {/* TITLE */}

      <h1 className="text-5xl font-bold mb-8 text-center">

        AI Driver Drowsiness Detection System

      </h1>

      {/* VIDEO FEED */}

      <div className="bg-white p-4 rounded-2xl shadow-xl">

        <img
          src="http://localhost:5000/video_feed"
          alt="Live Camera Feed"
          className="w-[850px] rounded-xl border-4 border-black"
        />

      </div>

      {/* STATUS */}

      <div
        className={`mt-8 px-10 py-5 rounded-2xl shadow-xl text-white text-3xl font-bold transition-all duration-300 ${
          status === "Drowsy"
            ? "bg-red-600"
            : "bg-green-600"
        }`}
      >

        Driver Status: {status}

      </div>

      {/* FEATURES */}

      <div className="mt-8 bg-white w-[850px] p-6 rounded-2xl shadow-lg">

        <h2 className="text-3xl font-bold mb-4">

          System Features

        </h2>

        <ul className="list-disc pl-8 text-lg space-y-3">

          <li>Real-time AI Driver Monitoring</li>

          <li>TensorFlow/Keras Drowsiness Detection</li>

          <li>Automatic Alarm Activation</li>

          <li>Twilio SMS Emergency Alert</li>

          <li>Twilio Voice Call Alert</li>

          <li>Live Frontend Status Updates</li>

          <li>False Alarm Reduction Logic</li>

        </ul>

      </div>

    </div>
  );
}

export default LiveMonitoring;