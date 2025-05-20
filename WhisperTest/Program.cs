using NAudio.CoreAudioApi;
using NAudio.Wave;
using NAudio.Wave.Compression;
using System;
using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Net;
using System.Threading.Tasks;
using Whisper.net;
using Whisper.net.Ggml;

public class Program
{
    public static async Task Main(string[] args)
    {
        //https://huggingface.co/ggerganov/whisper.cpp/tree/main
        string[] fileNames = [
            "ggml-tiny.bin",
            "ggml-tiny-q8_0.bin",
            "ggml-tiny-q5_1.bin",
            "ggml-tiny.en.bin",
            "ggml-tiny.en-q8_0.bin",
            "ggml-tiny.en-q5_1.bin",
            "ggml-base.bin",
            "ggml-base-q8_0.bin",
            "ggml-base-q5_1.bin",
            "ggml-base.en.bin",
            "ggml-base.en-q8_0.bin",
            "ggml-base.en-q5_1.bin",
            "ggml-small.bin",
            "ggml-small-q8_0.bin",
            "ggml-small-q5_1.bin",
            "ggml-small.en.bin",
            "ggml-small.en-q8_0.bin",
            "ggml-small.en-q5_1.bin",
            "ggml-medium.bin",
            "ggml-medium-q8_0.bin",
            "ggml-medium-q5_0.bin",
            "ggml-medium.en.bin",
            "ggml-medium.en-q8_0.bin",
            "ggml-medium.en-q5_0.bin",
            "ggml-large-v3-turbo.bin",
            "ggml-large-v3-turbo-q8_0.bin",
            "ggml-large-v3-turbo-q5_0.bin",
            "ggml-large-v3.bin",
            "ggml-large-v3-q5_0.bin",
            "ggml-large-v2.bin",
            "ggml-large-v2-q8_0.bin",
            "ggml-large-v2-q5_0.bin",
            "ggml-large-v1.bin"
        ];

        var modelFileName = "";
        foreach (var fileName in fileNames)
        {
            if (File.Exists(fileName))
            {
                modelFileName = fileName;
                break;
            }
        }

        if (string.IsNullOrWhiteSpace(modelFileName))
            modelFileName = args.FirstOrDefault();

        if (string.IsNullOrWhiteSpace(modelFileName) || !File.Exists(modelFileName))
        {
            Console.WriteLine(@"模型不存在！可以使用：“.\WhisperTest.exe xxx.bin”执行选择自定义模型");
            return;
        }

        using var whisperFactory = WhisperFactory.FromPath(modelFileName);

        Console.WriteLine($"选择模型：{Path.GetFileName(modelFileName)}");

        using var processor = whisperFactory.CreateBuilder()
            .WithLanguage("auto")
            //.WithPrompt("这是一段普通话聊天记录，是正常的说话顺序。")
            .Build();

        Console.WriteLine("开始录音和识别。。。按 Ctrl+C 可停止");

        RecordAndProcessAudio(processor);

        await Task.Delay(-1);
    }

    private static void RecordSystemAndProcessAudio(WhisperProcessor processor)
    {
        var _dataQueue = new List<byte>();
        var _dataLock = new object();

        var targetFormat = new WaveFormat(16000, 16, 1);
        var captureDevice = new MMDeviceEnumerator().GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);
        var capture = new WasapiLoopbackCapture(captureDevice);

        capture.DataAvailable += (s, a) =>
        {
            lock (_dataLock)
            {
                _dataQueue.AddRange(a.Buffer[..a.BytesRecorded]);
            }
        };
        capture.StartRecording();

        var t = new Thread(async () =>
        {
            while (true)
            {
                await Task.Delay(5000);
                var data = new List<byte>();
                lock (_dataLock)
                {
                    data.AddRange(_dataQueue);
                    _dataQueue.Clear();
                }
                if (data.Count == 0)
                {
                    Console.WriteLine($"没有声音...");
                    continue;
                }

                using var memoryStream = new MemoryStream();
                using var writer = new WaveFileWriter(memoryStream, capture.WaveFormat);
                writer.Write(data.ToArray(), 0, data.Count);
                writer.Flush();
                memoryStream.Seek(0, SeekOrigin.Begin);

                using WaveFileReader reader = new WaveFileReader(memoryStream);
                using var resampler = new MediaFoundationResampler(reader, targetFormat)
                {
                    ResamplerQuality = 60,
                };
                using var memoryStream2 = new MemoryStream();
                using WaveFileWriter waveFileWriter = new WaveFileWriter(memoryStream2, resampler.WaveFormat);
                var array = new byte[resampler.WaveFormat.AverageBytesPerSecond * 4];
                while (true)
                {
                    int num = resampler.Read(array, 0, array.Length);
                    if (num == 0)
                    {
                        break;
                    }

                    waveFileWriter.Write(array, 0, num);
                }
                waveFileWriter.Flush();
                memoryStream2.Seek(0, SeekOrigin.Begin);

                try
                {
                    Console.WriteLine($"识别中...");
                    await foreach (var result in processor.ProcessAsync(memoryStream2))
                    {
                        Console.WriteLine($"{result.Start}->{result.End}: {result.Text}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error processing audio: {ex.Message}");
                }
            }
        });
        t.Start();
    }

    private static void RecordAndProcessAudio(WhisperProcessor processor)
    {
        var cacheData = new List<byte>();
        var mode = 0;
        var hTime = DateTime.Now;
        var lTime = DateTime.Now;
        var pcacheData = new List<byte>();
        var startDate = DateTimeOffset.Now;
        var waveIn = new WaveInEvent
        {
            WaveFormat = new WaveFormat(16000, 16, 1) // 16kHz sample rate, mono
        };
        waveIn.DataAvailable += (s, a) =>
        {
            var buffer = a.Buffer[..a.BytesRecorded];
            if (buffer.Length == 0)
            {
                Console.WriteLine($"没有声音...");
                return;
            }
            float volume = CalculateVolume(buffer);
            pcacheData.AddRange(buffer);
            if (pcacheData.Count > 16000)
                pcacheData.Clear();
            if (volume > 0.01f)
            {
                hTime = DateTime.Now;
            }
            else
            {
                lTime = DateTime.Now;
            }
            if (mode == 0)
            {
                if (volume > 0.01f)
                {
                    Console.WriteLine("开始录音模式...");
                    cacheData.AddRange(pcacheData);
                    //cacheData.AddRange(a.Buffer);
                    mode = 1;
                }
            }
            else if (mode == 1)
            {
                cacheData.AddRange(buffer);
                Console.WriteLine("录音写入模式...");
                if ((lTime - hTime).TotalSeconds > 1)
                {
                    Console.WriteLine("结束录音模式...");
                    Task.Run(async () =>
                    {
                        mode = 2;
                        using var memoryStream = new MemoryStream();
                        using (var writer = new WaveFileWriter(memoryStream, waveIn.WaveFormat))
                        {
                            writer.Write(cacheData.ToArray(), 0, cacheData.Count);
                            writer.Flush();
                            memoryStream.Seek(0, SeekOrigin.Begin);

                            try
                            {
                                await foreach (var result in processor.ProcessAsync(memoryStream))
                                {
                                    Console.WriteLine($"{result.Start}->{result.End}: {result.Text}");
                                }
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Error processing audio: {ex.Message}");
                            }
                        }
                        cacheData.Clear();
                        mode = 0;
                    });
                }
            }
            else if (mode == 2)
            {
                Console.WriteLine("录音处理模式...");
            }
        };

        waveIn.StartRecording();
    }

    private static float CalculateVolume(byte[] buffer)
    {
        var bytesPerSample = 2; // 16位（2字节）每个样本
        var sampleCount = buffer.Length / bytesPerSample;
        var sum = 0.0f;

        for (var i = 0; i < sampleCount; i++)
        {
            var sample = BitConverter.ToInt16(buffer, i * bytesPerSample);
            sum += Math.Abs(sample) / 32768f; // 归一化到0-1范围
        }

        return sum / sampleCount;
    }
}
