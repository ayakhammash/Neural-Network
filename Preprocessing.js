import { Parser } from 'xml2js';

const parser = new Parser();

function preprocessInput(fruitData, uniqueClasses) {
    if (!Array.isArray(fruitData)) {
        console.error("Invalid input: fruitData is not an array");
        return [];
    }

    const maxSweetness = Math.max(...fruitData.map(fruit => fruit.Attributes.Sweetness)); //max values
   
    let matrix = [];
    fruitData.forEach(fruit => {
        if (!fruit.Attributes || !fruit.Fruit_Type) {
            console.error("Invalid fruit data format:", fruit);
            return;
        }

        let normalizedSweetness = fruit.Attributes.Sweetness / maxSweetness;
     
        let numericColor = convertColorToNumeric(fruit.Attributes.Color);

        let classOutput = convertClassToOutput(fruit.Fruit_Type, uniqueClasses);

        let row = [normalizedSweetness, numericColor, ...classOutput]; 
       // Spread operator to flatten the array
        matrix.push(row);
    });

    return matrix;
}

function convertColorToNumeric(color) {
    const colorMap = {'red': 1, 'green': 2, 'yellow': 3, 'orange': 4};
    return colorMap[color.toLowerCase()] || 0;
}

function convertClassToOutput(classLabel, uniqueClasses) {
    let output = new Array(uniqueClasses.length).fill(0);
    const index = uniqueClasses.indexOf(classLabel);
    if (index !== -1) {
        output[index] = 1;
    }
    return output;
}

export {preprocessInput}