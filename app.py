from flask import Flask, render_template
from flask import jsonify
from flask_cors import CORS
from flask import request
import re
from datetime import datetime

from pymongo import MongoClient
from simulation import Simulation

# create a Flask instance in our main module, and allow Cross-origin resource sharing
app = Flask(__name__)
CORS(app)

# todo try exception connection MongoDB
# connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['hydrolight']
collection = db["prova"]



@app.route("/")
def index():
    """
    router for root path "/"
    :return: HTML -> index.html
    """
    return render_template("dist/index.html")


@app.route("/lake", methods=['GET'])
def get_all_lakes():
    """
    router for lake.
    process different request: bottom, data, sensor_data.
    ff there is not request, response name and pathname of each output simulation file
    :return: JSON with request information
    """
    request_bottom = request.args.get('bottom')
    request_data = request.args.get('data')
    request_sensor_data = request.args.get('sensor_data')
    request_sensor_data_error = request.args.get('sensor_data_error')
    request_double_sensor_data_error = request.args.get('sensor_double_data_error')
    request_error = request.args.get('error')
    request_cluster = request.args.get('cluster')

    if request_bottom:
        # http://127.0.0.1:5000/lake?bottom=<request_bottom>
        regex = re.compile(request_bottom, re.IGNORECASE)
        output = []

        # set projection that specifies which fields MongoDB returns in the result set.
        for s in collection.find({'bottom': regex}, {"output": 0}):
            output.append(
                {'name': s['name'], 'pathname': s['pathname'], 'bottom': s['bottom'], 'depth': s['depth'],
                 'chl': s['chl'], 'cdom': s['cdom'], 'mineral': s['mineral'], 'cloud': s['cloud'],
                 'suntheta': s['suntheta'], 'windspeed': s['windspeed'], 'temp': s['temp'], 'salinity': s['salinity'],
                 'iop': s['iop']})
        return jsonify(output)

    elif request_data:
        # http://127.0.0.1:5000/lake?data=<request_data>
        if request_data == "all":
            file_name = datetime.now().strftime('%Y%m%d_%H%M%S')
            # this process is long, so we print to know that the process is started
            print("access to request data all")

            # set index to dataframe as name or pathname (it depends in which line is commented)
            output_index = []
            index = collection.find({}, {'bottom': 0, 'depth': 0, 'chl': 0, 'cdom': 0, 'mineral': 0, 'cloud': 0,
                                         'suntheta': 0, 'windspeed': 0, 'temp': 0, 'salinity': 0, 'iop': 0, 'input': 0,
                                         "output": 0})
            for i in index:
                # output_index.append(int(i['name']))
                output_index.append(i['pathname'])
            sim.set_index(output_index)

            # obtain name, pathname and output
            output = []
            resultat = collection.find({}, {'depth': 0, 'chl': 0, 'cdom': 0, 'mineral': 0, 'cloud': 0, 'suntheta': 0,
                                            'windspeed': 0, 'temp': 0, 'salinity': 0, 'iop': 0, 'input': 0})

            # for each name or pathname we want process the output file
            for num, s in enumerate(resultat, start=1):
                output.append(
                    {'name': s['name'], 'pathname': s['pathname'], 'output': s['output']})

                # process with name or pathname, it depends on the line is commented
                # sim.process_all_output_file(s['name'], s['output'])
                sim.process_all_output_file(s['pathname'], s['output'], file_name)
                # print to know how long the process. Print every 100 simulation has been done
                if (num % 50) == 0:
                    print(num)

            process_output = {'prova': 'prova'}

            return jsonify(process_output)

        else:
            s = collection.find_one({'name': request_data})
            output_file = s['output']
            output = sim.process_output_file(output_file)
            return jsonify(output)

    elif request_sensor_data:
        # http://127.0.0.1:5000/lake?sensor_data=<request_sensor_data>
        if request_sensor_data == "all":
            file_name = datetime.now().strftime('%Y%m%d_%H%M%S')
            # this process is long, so we print to know that the process is started
            print("access to request sensor_data all")

            # set index to dataframe as name or pathname (it depends in which line is commented)
            output_index = []
            index = collection.find({}, {'bottom': 0, 'depth': 0, 'chl': 0, 'cdom': 0, 'mineral': 0, 'cloud': 0,
                                         'suntheta': 0, 'windspeed': 0, 'temp': 0, 'salinity': 0, 'iop': 0, 'input': 0,
                                         "output": 0})
            for i in index:
                # output_index.append(int(i['name']))
                output_index.append(i['pathname'])

            # set index to Dataframe
            sim.set_index_sensor(output_index)

            # obtain name, pathname and output
            output = []
            resultat = collection.find({}, {'depth': 0, 'chl': 0, 'cdom': 0, 'mineral': 0, 'cloud': 0, 'suntheta': 0,
                                            'windspeed': 0, 'temp': 0, 'salinity': 0, 'iop': 0, 'input': 0})

            # for each name or pathname we want process the output file
            for num, s in enumerate(resultat, start=1):
                output.append(
                    {'name': s['name'], 'pathname': s['pathname'], 'output': s['output']})

                # process with name or pathname, it depends on the line is commented
                #sim.process_sensor_output_file(s['name'], s['output'], file_name)
                sim.process_sensor_output_file(s['pathname'], s['output'], file_name)
                # print to know how long the process. Print every 100 simulation has been done
                if (num % 50) == 0:
                    print(num)

            process_output = {'prova': 'prova'}
            return jsonify(process_output)

        elif request_sensor_data == "4":
            file_name = datetime.now().strftime('%Y%m%d_%H%M%S')
            # this process is long, so we print to know that the process is started
            print("access to request sensor_data all=4")

            # set index to dataframe as name or pathname (it depends in which line is commented)
            output_index = []
            index = collection.find({}, {'bottom': 0, 'depth': 0, 'chl': 0, 'cdom': 0, 'mineral': 0, 'cloud': 0,
                                         'suntheta': 0, 'windspeed': 0, 'temp': 0, 'salinity': 0, 'iop': 0, 'input': 0,
                                         "output": 0})
            for i in index:
                #output_index.append(int(i['name']))
                output_index.append(i['pathname'])

            # set index to Dataframe
            sim.set_index_sensor(output_index)

            # obtain name, pathname and output
            output = []
            resultat = collection.find({}, {'depth': 0, 'chl': 0, 'cdom': 0, 'mineral': 0, 'cloud': 0, 'suntheta': 0,
                                            'windspeed': 0, 'temp': 0, 'salinity': 0, 'iop': 0, 'input': 0})

            # for each name or pathname we want process the output file
            for num, s in enumerate(resultat, start=1):
                output.append(
                    {'name': s['name'], 'pathname': s['pathname'], 'output': s['output']})

                # process with name or pathname, it depends on the line is commented
                #sim.process_sensor_output_file(s['name'], s['output'], file_name)
                sim.process_sensor_z_output_file(s['pathname'], s['output'], file_name)
                # print to know how long the process. Print every 100 simulation has been done
                if (num % 50) == 0:
                    print(num)
            process_output = {'request_sensor_data': '4'}

            return jsonify(process_output)

    elif request_sensor_data_error and request_error:
        error = request_error
        if request_sensor_data_error == "4":
            file_name = datetime.now().strftime('%Y%m%d_%H%M%S')
            # this process is long, so we print to know that the process is started
            print("access to request sensor_data_error all=4")

            # set index to dataframe as name or pathname (it depends in which line is commented)
            output_index = []
            index = collection.find({}, {'bottom': 0, 'depth': 0, 'chl': 0, 'cdom': 0, 'mineral': 0, 'cloud': 0,
                                         'suntheta': 0, 'windspeed': 0, 'temp': 0, 'salinity': 0, 'iop': 0, 'input': 0,
                                         "output": 0})
            for i in index:
                #output_index.append(int(i['name']))
                output_index.append(i['pathname'])

            # set index to Dataframe
            sim.set_index_sensor(output_index)

            # obtain name, pathname and output
            output = []
            resultat = collection.find({}, {'depth': 0, 'chl': 0, 'cdom': 0, 'mineral': 0, 'cloud': 0, 'suntheta': 0,
                                            'windspeed': 0, 'temp': 0, 'salinity': 0, 'iop': 0, 'input': 0})

            # for each name or pathname we want process the output file
            for num, s in enumerate(resultat, start=1):
                output.append(
                    {'name': s['name'], 'pathname': s['pathname'], 'output': s['output']})

                # process with name or pathname, it depends on the line is commented
                #sim.process_sensor_output_file(s['name'], s['output'], file_name)
                sim.process_sensor_z_error_output_file(s['pathname'], s['output'], error, file_name)
                # print to know how long the process. Print every 100 simulation has been done
                if (num % 50) == 0:
                    print(num)
            process_output = {'request_sensor_data': '4'}

            return jsonify(process_output)

        else:
            # set index to Dataframe
            file_name = datetime.now().strftime('%Y%m%d_%H%M%S')
            s = collection.find_one({'name': request_sensor_data})
            sim.set_index_sensor([s['pathname']])

            output = sim.process_sensor_output_file(s['pathname'], s['output'], file_name)
            return jsonify(output)

    elif request_double_sensor_data_error and request_error:
        error = request_error
        if request_double_sensor_data_error == "4":
            file_name = datetime.now().strftime('%Y%m%d_%H%M%S')
            # this process is long, so we print to know that the process is started
            print("access to request double_sensor_data_error all=4")

            # set index to dataframe as name or pathname (it depends in which line is commented)
            output_index = []
            index = collection.find({}, {'bottom': 0, 'depth': 0, 'chl': 0, 'cdom': 0, 'mineral': 0, 'cloud': 0,
                                         'suntheta': 0, 'windspeed': 0, 'temp': 0, 'salinity': 0, 'iop': 0, 'input': 0,
                                         "output": 0})
            for i in index:
                # output_index.append(int(i['name']))
                output_index.append(i['pathname'])

            # set index to Dataframe
            sim.set_index_sensor(output_index)

            # obtain name, pathname and output
            output = []
            resultat = collection.find({}, {'depth': 0, 'chl': 0, 'cdom': 0, 'mineral': 0, 'cloud': 0, 'suntheta': 0,
                                            'windspeed': 0, 'temp': 0, 'salinity': 0, 'iop': 0, 'input': 0})

            # for each name or pathname we want process the output file
            for num, s in enumerate(resultat, start=1):
                output.append(
                    {'name': s['name'], 'pathname': s['pathname'], 'output': s['output']})

                # process with name or pathname, it depends on the line is commented
                # sim.process_sensor_output_file(s['name'], s['output'], file_name)
                sim.process_sensor_double_z_error_output_file(s['pathname'], s['output'], error, file_name)
                # print to know how long the process. Print every 100 simulation has been done
                if (num % 50) == 0:
                    print(num)
            process_output = {'request_double_sensor_data': '4'}

            return jsonify(process_output)

        else:
            # set index to Dataframe
            '''
            file_name = datetime.now().strftime('%Y%m%d_%H%M%S')
            s = collection.find_one({'name': request_sensor_data})
            sim.set_index_sensor([s['pathname']])

            output = sim.process_sensor_output_file(s['pathname'], s['output'], file_name)
            return jsonify(output)
            '''
            process_output = {'request_double_sensor_data': '4', 'depth' : 'null'}

            return jsonify(process_output)


    elif request_cluster:
        if request_cluster == "all":
            sim.cluster_all()
            return jsonify("cluster_all")

        elif request_cluster == "all_sensor":
            sim.cluster_all_sensor()
            return jsonify("cluster_all_sensor")

        elif request_cluster == "distances_sensor":
            sim.cluster_distances_sensor()
            return jsonify("cluster_distances_sensor")

        elif request_cluster == "distances_sensor_error_20":
            sim.cluster_distances_sensor_error_20()
            return jsonify("cluster_distances_sensor_error_20")

        elif request_cluster == "distances_sensor_error_10":
            sim.cluster_distances_sensor_error_10()
            return jsonify("cluster_distances_sensor_error_10")

        elif request_cluster == "distances_double_sensor_error_20":
            sim.cluster_distances_double_sensor_error_20()
            return jsonify("cluster_distances_double_sensor_error_20")

        elif request_cluster == "distances_double_sensor_error_10":
            sim.cluster_distances_double_sensor_error_10()
            return jsonify("cluster_distances_double_sensor_error_10")


    else:
        # http://127.0.0.1:5000/lake
        output = []
        resultat = collection.find({}, {"input": 0, "output": 0})
        for s in resultat:
            output.append({'name': s['name'], 'pathname': s['pathname']})
        return jsonify(output)


@app.route("/lake/<name>", methods=['GET'])
def get_one_lake(name):
    """
    Get details about single simulation
    :param name: name of lake simulation
    :return: JSON with requested info
    """
    # http://127.0.0.1:5000/lake/<name>
    s = collection.find_one({'name': name})
    if s:
        output = ({'name': s['name'], 'pathname': s['pathname'], 'bottom': s['bottom'], 'depth': s['depth'],
                   'chl': s['chl'], 'cdom': s['cdom'], 'mineral': s['mineral'], 'cloud': s['cloud'],
                   'suntheta': s['suntheta'], 'windspeed': s['windspeed'], 'temp': s['temp'], 'salinity': s['salinity'],
                   'iop': s['iop']})
    else:
        output = "no such name"
    return jsonify(output)


@app.route("/db", methods=['GET'])
def get_db():
    """
    Get name of db
    :param name: name of db simulation
    :return: JSON with requested info
    """
    output = []
    collection = db.collection_names(include_system_collections=False)
    for collect in collection:
        output.append(collect)
    return jsonify(output)


@app.route("/db/<name>", methods=['GET'])
def get_one_db(name):
    """
    Get name of db
    :param name: name of db simulation
    :return: JSON with requested info
    """
    global collection
    collection= db[name]
    output = "db " + name
    return jsonify(output)


# next routes are necessary for Angular client.
# /favicon, inline. , main. , polyfills. , styles.
@app.route("/favicon")
def favicon():
    return render_template("dist/favicon.ico")


@app.route("/inline.318b50c57b4eba3d437b.bundle.js")
def inline():
    return render_template("dist/inline.318b50c57b4eba3d437b.bundle.js")


@app.route("/main.b5d665777bca00e403eb.bundle.js")
def main():
    return render_template("dist/main.b5d665777bca00e403eb.bundle.js")


@app.route("/polyfills.b6b2cd0d4c472ac3ac12.bundle.js")
def polyfills():
    return render_template("dist/polyfills.b6b2cd0d4c472ac3ac12.bundle.js")


@app.route("/styles.ac89bfdd6de82636b768.bundle.css")
def styles():
    return render_template("dist/styles.ac89bfdd6de82636b768.bundle.css")


@app.route("/scripts.969129c85afcc2746e9c.bundle.js")
def scripts():
    return render_template("dist/scripts.969129c85afcc2746e9c.bundle.js")


if __name__ == '__main__':
    print("consultar linea 110 y 127 de app.py para cambiar entre name y pathname")
    sim = Simulation()
    app.run()
