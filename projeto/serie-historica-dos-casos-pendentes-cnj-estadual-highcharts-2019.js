Highcharts.chart('container', {
    chart: {
        type: 'line'
    },
    title: {
        text: ''
    },
    xAxis: {
        categories: ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']
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
        },
        series: {
        		label: {
                enabled: false
            },
            dataLabels: {
            		allowOverlap: true
            }
        }
    },
    series: [ {
        name: "Casos pendentes",
          data: [
		    { y: 49.4, dataLabels: { style: {color: 'blue'} } }, 
            { y: 50.3, dataLabels: { style: {color: 'blue'} } }, 
            { y: 52.3, dataLabels: { style: {color: 'blue'} } }, 
            { y: 54.1, dataLabels: { style: {color: 'blue'} } }, 
            { y: 58.0, dataLabels: { style: {color: 'blue'} } }, 
            { y: 57.3, dataLabels: { style: {color: 'blue'} } }, 
            { y: 62.1, dataLabels: { style: {color: 'blue'} } }, 
            { y: 63.2, dataLabels: { style: {color: 'blue'} } }, 
            { y: 63.0, dataLabels: { style: {color: 'blue'} } }, 
            { y: 63.0, dataLabels: { style: {color: 'blue'} } }
		]
    }, {
        name: "Processos baixados",
          data: [18.3,17.1,18.1,19.1,19.4,19.9,20.1,20.8,21.4,22.3]
    }, {
        name: "Casos novos",
          data: [
          	{ y: 17.8, dataLabels: { x: 0, y: 30, style: {color: 'green'} } }, 
            { y: 17.5, dataLabels: { x: 0, y: 30, style: {color: 'green'} } }, 
            { y: 18.6, dataLabels: { x: 0, y: 30, style: {color: 'green'} } }, 
            { y: 19.8, dataLabels: { x: 0, y: 30, style: {color: 'green'} } }, 
            { y: 20.5, dataLabels: { x: 0, y: 30, style: {color: 'green'} } }, 
            { y: 20.3, dataLabels: { x: 0, y: 30, style: {color: 'green'} } }, 
            { y: 19.4, dataLabels: { x: 0, y: 30, style: {color: 'green'} } }, 
            { y: 19.8, dataLabels: { x: 0, y: 30, style: {color: 'green'} } }, 
            { y: 19.7, dataLabels: { x: 0, y: 30, style: {color: 'green'} } }, 
            { y: 19.6, dataLabels: { x: 0, y: 30, style: {color: 'green'} } }
          ]
    }]
});