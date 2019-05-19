Highcharts.chart('container', {
    chart: {
        type: 'line'
    },
    title: {
        text: ''
    },
    xAxis: {
        categories: ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
    },
    yAxis: {
        title: {
            text: 'Milhões'
        }
    },
    plotOptions: {
        line: {
            dataLabels: {
                enabled: true
            },
            enableMouseTracking: false
        }
    },
    series: [{
    		name: "Casos pendentes",
        data: [60.7,61.9,64.4,67.1,71.6,72.0,76.9,79.8,80.1]
    }]
});