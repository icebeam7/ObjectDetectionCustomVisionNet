using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Training.Models;
using Microsoft.Azure.CognitiveServices.Vision.CustomVision.Prediction;

namespace ObjectDetectionCustomVisionNet
{
    class Program
    {
        static string trainingFolder = @"/Users/luisbeltran/Desktop/CustomVision/Dataset/test";

        public static async Task Main(string[] args)
        {
            // Project name must match the one in Custom Vision
            var projectName = "Open Images Detector";
            var publishedModelName = "OpenImagesDetectorModel";

            // Replace with your own from Custom Vision project
            var customVisionKey = "";
            var endpoint = "";
            var resourceId = "";

            if (string.IsNullOrWhiteSpace(endpoint) ||
                string.IsNullOrWhiteSpace(customVisionKey) ||
                string.IsNullOrWhiteSpace(resourceId))
            {
                Console.WriteLine("You need to set the endpoint, key and resource id. The program will end;");
                Console.ReadKey();
                return;
            }

            // Training Client
            var trainingApi = new CustomVisionTrainingClient()
            {
                ApiKey = customVisionKey,
                Endpoint = endpoint
            };

            Console.WriteLine($"----- Selecting existing project: {projectName}... -----");

            var projects = await trainingApi.GetProjectsAsync();
            var project = projects.FirstOrDefault(x => x.Name == projectName);

            if (project != null)
            {
                Console.WriteLine($"\t{projectName} found in Custom Vision workspace.");

                var domainId = project.Settings.DomainId;
                var projectDomain = await trainingApi.GetDomainAsync(domainId);

                Console.WriteLine($"\tProject domain: {projectDomain.Name}.");
                Console.WriteLine($"\tProject type: {projectDomain.Type}.");

                WriteSeparator();
            }
            else
            {
                Console.WriteLine($"\tProject {projectName} was not found in your subscription. The program will end.");
                return;
            }

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();

            // Retrieve the tags that already exist in the project
            Console.WriteLine("----- Retrieving tags... -----");

            var modelTags = await trainingApi.GetTagsAsync(project.Id);

            var tags = new List<Tag>();
            var onlineImages = 0;

            // Obtain tags from our dataset
            var tagsFile = Path.Combine(trainingFolder, "tags.txt");
            var imageLabels = await File.ReadAllLinesAsync(tagsFile);

            foreach (var label in imageLabels)
            {
                // Check if the label already exists
                var tag = modelTags.FirstOrDefault(x => x.Name == label);

                if (tag == null)
                {
                    // If not, create it
                    tag = await trainingApi.CreateTagAsync(project.Id, label);

                    Console.WriteLine($"\tTag {tag.Name} was created.");
                }
                else
                {
                    // If so, just count images with this tag
                    onlineImages += tag.ImageCount;
                    Console.WriteLine($"\tTag {label} was NOT created (it already exists)");
                }

                tags.Add(tag);
            }

            WriteSeparator();
            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();

            // Upload images
            var uploadImages = true;

            if (onlineImages > 0)
            {
                Console.WriteLine($"There are {onlineImages} training images already uploaded. Do you want to upload more? (Y/N)");
                uploadImages = Console.ReadKey().Key == ConsoleKey.Y;
            }

            Iteration iteration = null;

            if (uploadImages)
            {
                Console.WriteLine("----- Accessing images -----");
                var imageFileEntries = new List<ImageFileCreateEntry>();

                foreach (var label in imageLabels)
                {
                    var tagFolder = Path.Combine(trainingFolder, label);
                    var tagImages = Directory.GetFiles(tagFolder);
                    var tagLabels = Path.Combine(tagFolder, "normalizedLabel");

                    foreach (var image in tagImages)
                    {
                        var imageName = Path.GetFileNameWithoutExtension(image);
                        var imageFile = Path.GetFileName(image);
                        var imageLabelFile = $"{imageName}.txt";
                        var tagCV = tags.Single(x => x.Name == label);

                        var imageBoundingBoxes = await File.ReadAllLinesAsync(Path.Combine(tagLabels, imageLabelFile));
                        var imageRegions = new List<Region>();

                        foreach (var bbox in imageBoundingBoxes)
                        {
                            var normalizedData = bbox.Split();
                            var left = float.Parse(normalizedData[0]);
                            var top = float.Parse(normalizedData[1]);
                            var width = float.Parse(normalizedData[2]);
                            var height = float.Parse(normalizedData[3]);

                            imageRegions.Add(new Region(tagCV.Id, left, top, width, height));
                        }

                        Console.WriteLine($"\tAdding image {imageName} with its regions.");

                        imageFileEntries.Add(new ImageFileCreateEntry(
                            imageFile, await File.ReadAllBytesAsync(image), null, imageRegions));
                    }
                }

                var batchPageSize = 64;
                var numBatches = imageFileEntries.Count / batchPageSize;
                var sizeLastBatch = imageFileEntries.Count % batchPageSize;

                for (int batch = 0; batch < numBatches; batch++)
                {
                    Console.WriteLine($"\tUploading images batch #{batch}.");

                    var entries = imageFileEntries.Skip(batch * batchPageSize).Take(batchPageSize).ToList();

                    await trainingApi.CreateImagesFromFilesAsync(
                        project.Id, new ImageFileCreateBatch(entries));
                }

                if (sizeLastBatch > 0)
                {
                    Console.WriteLine($"\tUploading last batch.");
                    var lastEntries = imageFileEntries.Skip(numBatches * batchPageSize).Take(sizeLastBatch).ToList();

                    await trainingApi.CreateImagesFromFilesAsync(project.Id,
                        new ImageFileCreateBatch(lastEntries));
                }

                WriteSeparator();
                Console.WriteLine("Press any key to continue...");
                Console.ReadKey();

                try
                {
                    // Now there are images with tags start training the project
                    Console.WriteLine("----- Starting the Training process... -----");

                    iteration = await trainingApi.TrainProjectAsync(project.Id);

                    // The returned iteration will be in progress, and can be queried periodically to see when it has completed
                    while (iteration.Status == "Training")
                    {
                        Thread.Sleep(1000);
                        Console.WriteLine($"\tIteration '{iteration.Name}' status: {iteration.Status}");

                        // Re-query the iteration to get it's updated status
                        iteration = await trainingApi.GetIterationAsync(project.Id, iteration.Id);
                    }

                    Console.WriteLine($"\tIteration '{iteration.Name}' status: {iteration.Status}");

                    WriteSeparator();
                    Console.WriteLine("Press any key to continue...");
                    Console.ReadKey();

                    // The iteration is now trained. Publish it to the prediction endpoint.
                    Console.WriteLine($"----- Starting the Publication process. -----");

                    await trainingApi.PublishIterationAsync(
                        project.Id, iteration.Id, publishedModelName, resourceId);

                    Console.WriteLine($"\tIteration '{iteration.Name}' published.");
                    WriteSeparator();
                    Console.WriteLine("Press any key to continue...");
                    Console.ReadKey();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"There was an exception (perhaps nothing changed since last iteration?).");
                }
            }

            if (iteration == null)
            {
                var iterations = await trainingApi.GetIterationsAsync(project.Id);

                iteration = iterations.OrderByDescending(x => x.LastModified).FirstOrDefault();

                Console.WriteLine($"Iteration '{iteration.Name}' found and loaded.");
                WriteSeparator();
                Console.WriteLine("Press any key to continue...");
                Console.ReadKey();
            }

            // Prediction Client
            var predictionClient = new CustomVisionPredictionClient()
            {
                ApiKey = customVisionKey,
                Endpoint = endpoint
            };

            // Make predictions against the new project
            Console.WriteLine("----- Making predictions -----");
            var testImages = LoadImagesFromDisk("Test");

            foreach (var image in testImages)
            {
                var imageName = Path.GetFileName(image);

                using (var stream = new MemoryStream(File.ReadAllBytes(image)))
                {
                    Console.WriteLine($"\tImage: {imageName}");

                    var result = await predictionClient.DetectImageAsync(
                        project.Id, publishedModelName, stream);

                    // Loop over each prediction and write out the results
                    foreach (var prediction in result.Predictions.OrderByDescending(x => x.Probability))
                    {
                        Console.WriteLine($"\t\tFor Tag '{prediction.TagName}': {prediction.Probability:P3} " +
                            $"[{prediction.BoundingBox.Left}, {prediction.BoundingBox.Top}, " +
                            $"{prediction.BoundingBox.Width}, {prediction.BoundingBox.Height}]");
                    }

                    WriteSeparator();
                    Console.WriteLine("Press any key for next image...");
                    Console.ReadKey();
                }
            }

            WriteSeparator();

            Console.WriteLine("----- Do you want to export the model? (Y/N) -----");
            var exportModel = Console.ReadKey().Key == ConsoleKey.Y;

            if (exportModel)
            {
                do
                {
                    var platform = string.Empty;
                    var extension = string.Empty;
                    Export export;

                    Console.WriteLine("\tOptions: \n\t1) TensorFlow \n\t2) CoreML \n\t3) Other platform \n\tE) End program");
                    var option = Console.ReadKey().Key;

                    switch (option)
                    {
                        case ConsoleKey.D1:
                            platform = "TensorFlow";
                            extension = "zip";
                            break;
                        case ConsoleKey.D2:
                            platform = "CoreML";
                            extension = "mlmodel";
                            break;
                        case ConsoleKey.D3:
                            Console.WriteLine("\tType the platform name");
                            platform = Console.ReadLine();
                            Console.WriteLine($"\tNow type the file extension for the {platform} exported model.");
                            extension = Console.ReadLine();
                            break;
                        case ConsoleKey.E:
                            exportModel = false;
                            break;
                        default:
                            Console.WriteLine("\n\tOption not supported.");
                            break;
                    }

                    WriteSeparator();

                    if (!string.IsNullOrWhiteSpace(platform))
                    {
                        try
                        {
                            Console.WriteLine($"\tExporting to {platform}...");

                            do
                            {
                                var exports = await trainingApi.GetExportsAsync(project.Id, iteration.Id);

                                export = exports.FirstOrDefault(x => x.Platform == platform);

                                if (export == null)
                                    export = await trainingApi.ExportIterationAsync(project.Id, iteration.Id, platform);

                                Thread.Sleep(1000);
                                Console.WriteLine($"\tStatus: {export.Status}");
                            } while (export.Status == "Exporting");

                            Console.WriteLine($"Status: {export.Status}");

                            if (export.Status == ExportStatus.Done)
                            {
                                Console.WriteLine($"\tDownloading {platform} model");
                                var filePath = Path.Combine(Environment.CurrentDirectory, $"{publishedModelName}_{platform}.{extension}");

                                using (var httpClient = new HttpClient())
                                {
                                    using (var stream = await httpClient.GetStreamAsync(export.DownloadUri))
                                    {
                                        using (var file = new FileStream(filePath, FileMode.Create))
                                        {
                                            await stream.CopyToAsync(file);
                                            Console.WriteLine($"\tModel successfully exported. You can find it here:\n\t{filePath}.");
                                            WriteSeparator();
                                        }
                                    }
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Exception found: {ex.Message}");
                            WriteSeparator();
                        }
                    }
                } while (exportModel);
            }

            Console.WriteLine("Press a key to exit the program!");
            Console.ReadKey();
        }

        private static List<string> LoadImagesFromDisk(string directory) =>
            Directory.GetFiles(directory).ToList();

        private static string RepeatCharacter(string s, int n) =>
            new StringBuilder(s.Length * n).AppendJoin(s, new string[n + 1]).ToString();

        private static void WriteSeparator() =>
            Console.WriteLine(RepeatCharacter("-", 30));
    }
}