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
    series: [{
    		name: "Casos pendentes",
			data: [
				{ y: 60.7, dataLabels: { style: {color: 'blue'} } }, 
				{ y: 61.9, dataLabels: { style: {color: 'blue'} } }, 
				{ y: 64.4, dataLabels: { style: {color: 'blue'} } }, 
				{ y: 67.1, dataLabels: { style: {color: 'blue'} } }, 
				{ y: 71.6, dataLabels: { style: {color: 'blue'} } }, 
				{ y: 72.0, dataLabels: { style: {color: 'blue'} } }, 
				{ y: 77.1, dataLabels: { style: {color: 'blue'} } }, 
				{ y: 79.8, dataLabels: { style: {color: 'blue'} } }, 
				{ y: 79.6, dataLabels: { style: {color: 'blue'} } }, 
				{ y: 78.7, dataLabels: { style: {color: 'blue'} } }
			]
    }, {
    		name: "Processos baixados",
        	data: [25.3,24.1,25.8,27.7,28.1,28.4,28.6,29.5,30.7,31.9]
    }, {
    		name: "Casos novos",
          data: [
          	{ y: 24.6, dataLabels: { x: 0, y: 20, style: {color: 'green'} } }, 
            { y: 24.0, dataLabels: { x: 0, y: 20, style: {color: 'green'} } }, 
            { y: 26.1, dataLabels: { x: 0, y: 30, style: {color: 'green'} } }, 
            { y: 28.0, dataLabels: { x: 0, y: 30, style: {color: 'green'} } }, 
            { y: 28.5, dataLabels: { x: 0, y: 30, style: {color: 'green'} } }, 
            { y: 29.0, dataLabels: { x: 0, y: 30, style: {color: 'green'} } }, 
            { y: 27.8, dataLabels: { x: 0, y: 30, style: {color: 'green'} } }, 
            { y: 29.3, dataLabels: { x: 0, y: 30, style: {color: 'green'} } }, 
            { y: 28.6, dataLabels: { x: 0, y: 30, style: {color: 'green'} } }, 
            { y: 28.1, dataLabels: { x: 0, y: 30, style: {color: 'green'} } }
          ]
    }]
});