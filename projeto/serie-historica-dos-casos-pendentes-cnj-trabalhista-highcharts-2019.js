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
          	{ y: 3.3, dataLabels: { x: 0, y: 30, style: {color: 'lightblue'} } }, 
            { y: 3.4, dataLabels: { x: 0, y: -20, style: {color: 'lightblue'} } }, 
            { y: 3.4, dataLabels: { x: 0, y: 30, style: {color: 'lightblue'} } }, 
            { y: 4.0, dataLabels: { x: 0, y: -10, style: {color: 'lightblue'} } }, 
            { y: 4.5, dataLabels: { x: 0, y: 30, style: {color: 'lightblue'} } }, 
            { y: 4.6, dataLabels: { x: 0, y: -10, style: {color: 'lightblue'} } }, 
            { y: 5.1, dataLabels: { x: 0, y: 30, style: {color: 'lightblue'} } }, 
            { y: 5.4, dataLabels: { x: 0, y: 30, style: {color: 'lightblue'} } }, 
            { y: 5.5, dataLabels: { x: 0, y: 30, style: {color: 'lightblue'} } }, 
            { y: 4.9, dataLabels: { x: 0, y: 30, style: {color: 'lightblue'} } }
          ]
    }, {
    		name: "Processos baixados",
        	data: [3.3,3.4,3.7,3.8,4.0,4.2,4.3,4.2,4.5,4.4]
    }, {
    		name: "Casos novos",
          data: [
          	{ y: 3.4, dataLabels: { x: 0, y: -20, style: {color: 'lightgreen'} } }, 
            { y: 3.3, dataLabels: { x: 0, y: 30, style: {color: 'lightgreen'} } }, 
            { y: 3.7, dataLabels: { x: 0, y: -20, style: {color: 'lightgreen'} } }, 
            { y: 3.9, dataLabels: { x: 0, y: 30, style: {color: 'lightgreen'} } }, 
            { y: 4.0, dataLabels: { x: 0, y: 30, style: {color: 'lightgreen'} } }, 
            { y: 4.0, dataLabels: { x: 0, y: 30, style: {color: 'lightgreen'} } }, 
            { y: 4.1, dataLabels: { x: 0, y: 30, style: {color: 'lightgreen'} } }, 
            { y: 4.3, dataLabels: { x: 0, y: -10, style: {color: 'lightgreen'} } }, 
            { y: 4.3, dataLabels: { x: 0, y: 30, style: {color: 'lightgreen'} } }, 
            { y: 3.5, dataLabels: { x: 0, y: 30, style: {color: 'lightgreen'} } }
          ]
    }]
});