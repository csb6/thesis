const height = 800;
const width = 300;
const dsvParser = d3.dsvFormat(" ");

function get_data(dataString) {
    return dsvParser.parse(dataString, (row) => {
        row.orig_rank = parseInt(row.orig_rank);
        row.new_rank = parseInt(row.new_rank);
        return row;
    });
}

const tf_dataset = get_data(tf_data);
const tf_log_idf_log_dataset = get_data(tf_log_idf_log_data);
let curr_dataset = tf_dataset;

let plot = d3.select("#chart")
    .append("svg")
    .attr("width", width+30)
    .attr("height", height)
    .attr("id", "plot");

let rankScale = null;

const kindToColor = {
    unhealthy: "black",
    neutral: "blue",
    healthy: "green"
}

function plot_data(data, threshold) {
    rankScale = d3.scaleLinear()
        .domain([1, d3.max(data, (row) => row.orig_rank)])
        .range([1, height]);

    // Axes
    plot.selectAll("g").remove();
    plot.append("g")
        .attr("transform", "translate(50 0)")
        .call(d3.axisLeft(rankScale));

    plot.append("g")
        .attr("transform", `translate(${width} 0)`)
        .call(d3.axisRight(rankScale));

    function setup(selection) {
        selection
            .filter((row) => Math.abs(row.orig_rank - row.new_rank) > threshold)
            .attr("x1", 50)
            .attr("y1", (row) => rankScale(row.orig_rank))
            .attr("x2", width)
            .attr("y2", (row) => rankScale(row.new_rank))
            .attr("stroke", (row) => kindToColor[row.kind])
            .attr("stroke-width", 3)
            .on("mouseover", function(e, row) {
                let sel = d3.select(this);
                sel.style("opacity", 0.5);
                console.log(d3.select("#current-word"));
                console.log(row.word)
                d3.select("#current-word").text(row.word);
            })
            .on("mouseout", function(e, row) {
                let sel = d3.select(this);
                sel.style("opacity", 1.0);
                d3.select("#current-word").text("-");
            });
    }

    // Data
    plot.selectAll("line").remove();
    plot.selectAll("line")
        .data(data)
        .enter()
        .append("line")
        .call(setup);
}

plot_data(tf_dataset, 10)
curr_dataset = tf_dataset;

// Controls
d3.select("#threshold-input")
    .on("input", function(e) {
        plot_data(curr_dataset, parseInt(this.value));
    })

document.getElementById("unhealthy-button").onclick = function() {
    plot.selectAll("line")
        .filter((row) => row.kind === "unhealthy")
        .attr("stroke-width", 3);
    plot.selectAll("line")
        .filter((row) => row.kind !== "unhealthy")
        .attr("stroke-width", 0);
}

document.getElementById("neutral-button").onclick = function() {
    plot.selectAll("line")
        .filter((row) => row.kind === "neutral")
        .attr("stroke-width", 3);
    plot.selectAll("line")
        .filter((row) => row.kind !== "neutral")
        .attr("stroke-width", 0);
}

document.getElementById("healthy-button").onclick = function() {
    plot.selectAll("line")
        .filter((row) => row.kind === "healthy")
        .attr("stroke-width", 3);
    plot.selectAll("line")
        .filter((row) => row.kind !== "healthy")
        .attr("stroke-width", 0);
}

document.getElementById("all-button").onclick = function() {
    plot.selectAll("line")
        .attr("stroke-width", 3);
}

document.getElementById("tf-rank-button").onclick = function() {
    plot_data(tf_dataset, 10);
    d3.select("#threshold-input").attr("value", 10);
    curr_dataset = tf_dataset;
}

document.getElementById("tf-log-idf-log-rank-button").onclick = function() {
    plot_data(tf_log_idf_log_dataset, 75);
    d3.select("#threshold-input").attr("value", 75);
    curr_dataset = tf_dataset;
}

document.getElementById("lower-rank-button").onclick = function() {
    plot.selectAll("line")
        .filter((row) => row.new_rank - row.orig_rank > 0)
        .attr("stroke-width", 3);
    plot.selectAll("line")
        .filter((row) => row.new_rank - row.orig_rank <= 0)
        .attr("stroke-width", 0);
}

document.getElementById("higher-rank-button").onclick = function() {
    plot.selectAll("line")
        .filter((row) => row.new_rank - row.orig_rank <= 0)
        .attr("stroke-width", 3);
    plot.selectAll("line")
        .filter((row) => row.new_rank - row.orig_rank > 0)
        .attr("stroke-width", 0);
}
