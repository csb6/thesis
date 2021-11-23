const height = 800;
const width = 300;

let rankScale = d3.scaleLinear()
    .domain([1, d3.max(data, (row) => row.orig_rank)])
    .range([1, height]);

let plot = d3.select("#chart")
    .append("svg")
    .attr("width", width+30)
    .attr("height", height)
    .attr("id", "plot");

const kindToColor = {
    unhealthy: "black",
    neutral: "blue",
    healthy: "green"
}

// Axes
plot.append("g")
    .attr("transform", "translate(50 0)")
    .call(d3.axisLeft(rankScale));

plot.append("g")
    .attr("transform", `translate(${width} 0)`)
    .call(d3.axisRight(rankScale));

function setup(selection) {
    selection
        .filter((row) => Math.abs(row.orig_rank - row.new_rank) > 10)
        .attr("x1", 50)
        .attr("y1", (row) => rankScale(row.orig_rank))
        .attr("x2", width)
        .attr("y2", (row) => rankScale(row.new_rank))
        .attr("stroke", (row) => kindToColor[row.kind])
        .attr("stroke-width", 2);
}

// Data
plot.selectAll("line")
    .data(data)
    .enter()
    .append("line")
    .call(setup);

// Controls
document.getElementById("unhealthy-button").onclick = function() {
    plot.selectAll("line")
        .filter((row) => row.kind === "unhealthy")
        .attr("stroke-width", 2);
    plot.selectAll("line")
        .filter((row) => row.kind !== "unhealthy")
        .attr("stroke-width", 0);
}

document.getElementById("neutral-button").onclick = function() {
    plot.selectAll("line")
        .filter((row) => row.kind === "neutral")
        .attr("stroke-width", 2);
    plot.selectAll("line")
        .filter((row) => row.kind !== "neutral")
        .attr("stroke-width", 0);
}

document.getElementById("healthy-button").onclick = function() {
    plot.selectAll("line")
        .filter((row) => row.kind === "healthy")
        .attr("stroke-width", 2);
    plot.selectAll("line")
        .filter((row) => row.kind !== "healthy")
        .attr("stroke-width", 0);
}

document.getElementById("all-button").onclick = function() {
    plot.selectAll("line")
        .attr("stroke-width", 2);
}
