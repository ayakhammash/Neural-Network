import express from "express";
import multer from "multer"; //local storage
import cors from 'cors'; //connection
import { trainNetwork, predict, initializeNetwork } from "./NeuralNetwork.js";
import { preprocessInput } from "./Preprocessing.js";

const app = express();
const port = 3000;
const upload = multer({ storage: multer.memoryStorage() });
app.use(express.json());
app.use(cors());
app.use(express.static("Public"));

let trainingProgress = 0;
let trainedNetwork = null;

app.post('/train', async (req, res) => {
    try {
        const uniqueClasses = ['Apple', 'Banana', 'Orange'];
        const { fruitData, config } = req.body;
        const preprocessedData = preprocessInput(fruitData, uniqueClasses);
        const { numberOfNeurons, activationFunction, learningRate,epochsNumber } = req.body.config;
        const inputSize = preprocessedData.length;
        const outputSize = uniqueClasses.length;
        const hiddenSize = config.numberOfNeurons;
        const network = initializeNetwork(inputSize, hiddenSize, outputSize);
      // console.log(activationFunction);
//console.log(network);
        // console.log({ fruitData, config });
        //  console.log("Before",network);
        
        const trainingResults = await trainNetwork(
            preprocessedData, 
            network, 
            epochsNumber, 
            learningRate, 
            activationFunction, 
            uniqueClasses, 
        );
      
         trainedNetwork = trainingResults;
        // console.log(trainingResults);
        // res.json({ message: 'Training completed successfully', results: trainingResults });
    } catch (error) {
        console.error("Error during training:", error);
        res.status(500).json({ message: "An error occurred during training." });
    }
});



app.post("/test", express.json(), async (req, res) => { //checking the class of the data that the user sends
    try {
        const { color, sweetness } = req.body;
        if (!trainedNetwork) {
            return res.status(400).send("No trained model available.");
        }
        const prediction = predict(trainedNetwork, { color, sweetness });
        res.json({ prediction: prediction });
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: "An error occurred during prediction." });
    }
});

app.get("/training-progress", (req, res) => {
    res.json({ progress: trainingProgress });
});

app.listen(port, () => console.log(`Server running on port ${port}`));
