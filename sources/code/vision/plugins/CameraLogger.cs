using System.IO;
using BepInEx;
using UnityEngine;

namespace GoiCameraLogger
{
    [BepInPlugin("goi.cameralogger", "Camera Logger", "1.0.0")]
    public class CameraLoggerPlugin : BaseUnityPlugin
    {
        private Camera _mainCamera;
        private string _outputPath;
        private float _lastWriteTime;
        private const float WriteInterval = 0.05f;

        void Awake()
        {
            _outputPath = Path.Combine(Application.dataPath, "..", "camera_y.txt");
            Logger.LogInfo("CameraLogger started, output: " + _outputPath);
        }

        void Update()
        {
            if (_mainCamera == null)
            {
                _mainCamera = Camera.main;
                if (_mainCamera == null) return;
                Logger.LogInfo("Found camera: " + _mainCamera.name);
            }

            if (Time.time - _lastWriteTime >= WriteInterval)
            {
                _lastWriteTime = Time.time;
                Vector3 pos = _mainCamera.transform.position;
                string line = pos.x.ToString("F4") + "," + pos.y.ToString("F4") + "," + pos.z.ToString("F4");
                File.WriteAllText(_outputPath, line);
            }
        }
    }
}
